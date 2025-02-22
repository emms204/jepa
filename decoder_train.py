import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import wandb
import numpy as np
from tqdm import tqdm
import yaml

from app.vjepa.utils import (
    init_video_model,
    init_opt,
)

from src.utils.logging import get_logger
from src.datasets.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from app.vjepa.transforms import make_transforms
from src.utils.tensors import repeat_interleave_batch

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)

class VJEPADecoderTrainer:
    def __init__(
        self,
        config,
        device='cuda'
    ):
        
        # Setup device
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # -- META
        cfgs_meta = self.config.get('meta')
        pretrain_folder = cfgs_meta.get('folder', None)
        ckp_fname = cfgs_meta.get('checkpoint', None)
        use_sdpa = cfgs_meta.get('use_sdpa', False)
        which_dtype = cfgs_meta.get('dtype')
        if which_dtype.lower() == 'bfloat16':
            dtype = torch.bfloat16
            mixed_precision = True
        elif which_dtype.lower() == 'float16':
            dtype = torch.float16
            mixed_precision = True
        else:
            dtype = torch.float32
            mixed_precision = False

        pretrained_path = os.path.join(pretrain_folder, ckp_fname)

        # -- MASK
        cfgs_mask = self.config.get('mask')

        # -- MODEL
        cfgs_model = self.config.get('model')
        model_name = cfgs_model.get('model_name')
        pred_depth = cfgs_model.get('pred_depth')
        pred_embed_dim = cfgs_model.get('pred_embed_dim')
        uniform_power = cfgs_model.get('uniform_power', True)
        use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
        zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)

        # -- DATA
        cfgs_data = self.config.get('data')
        dataset_type = cfgs_data.get('dataset_type', 'videodataset')
        mask_type = cfgs_data.get('mask_type', 'multiblock3d')
        train_path = cfgs_data.get('train_csv_path')
        val_path = cfgs_data.get('val_csv_path')
        datasets_weights = cfgs_data.get('datasets_weights', None)
        batch_size = cfgs_data.get('batch_size')
        num_clips = cfgs_data.get('num_clips')
        num_frames = cfgs_data.get('num_frames')
        tubelet_size = cfgs_data.get('tubelet_size')
        sampling_rate = cfgs_data.get('sampling_rate')
        duration = cfgs_data.get('clip_duration', None)
        crop_size = cfgs_data.get('crop_size', 224)
        patch_size = cfgs_data.get('patch_size')
        pin_mem = cfgs_data.get('pin_mem', False)
        num_workers = cfgs_data.get('num_workers', 1)
        filter_short_videos = cfgs_data.get('filter_short_videos', False)
        decode_one_clip = cfgs_data.get('decode_one_clip', True)
        log_resource_util_data = cfgs_data.get('log_resource_utilization', False)

        # -- DATA AUGS
        cfgs_data_aug = self.config.get('data_aug')
        ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
        rr_scale = cfgs_data_aug.get('random_resize_scale', [0.3, 1.0])
        motion_shift = cfgs_data_aug.get('motion_shift', False)
        reprob = cfgs_data_aug.get('reprob', 0.)
        use_aa = cfgs_data_aug.get('auto_augment', False)

        encoder, predictor = init_video_model(
            uniform_power=uniform_power,
            use_mask_tokens=use_mask_tokens,
            num_mask_tokens=len(cfgs_mask),
            zero_init_mask_tokens=zero_init_mask_tokens,
            device=self.device,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            model_name=model_name,
            crop_size=crop_size,
            pred_depth=pred_depth,
            pred_embed_dim=pred_embed_dim,
            use_sdpa=use_sdpa,
        )
        
        # Models
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)

        self._load_checkpoint(pretrained_path)

        self.decoder = self._init_decoder()
        
        # Freeze encoder and predictor
        for model in [self.encoder, self.predictor]:
            for param in model.parameters():
                param.requires_grad = False

        # -- make data transforms
        if mask_type == 'multiblock3d':
            logger.info('Initializing basic multi-block mask')
            mask_collator = MB3DMaskCollator(
                crop_size=crop_size,
                num_frames=num_frames,
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                cfgs_mask=cfgs_mask)
        else:
            logger.info('Initializing random tube mask')
            mask_collator = TubeMaskCollator(
                crop_size=crop_size,
                num_frames=num_frames,
                patch_size=patch_size,
                tubelet_size=tubelet_size,
                cfgs_mask=cfgs_mask)
            
        transform = make_transforms(
            random_horizontal_flip=True,
            random_resize_aspect_ratio=ar_range,
            random_resize_scale=rr_scale,
            reprob=reprob,
            auto_augment=use_aa,
            motion_shift=motion_shift,
            crop_size=crop_size)


        
        # Initialize datasets and dataloaders
        train_dataset, train_loader = init_data(
            data='videodataset',
            root_path=train_path,
            batch_size=batch_size,
            training=True,
            clip_len=num_frames,
            frame_sample_rate=sampling_rate,
            filter_short_videos=filter_short_videos,
            decode_one_clip=decode_one_clip,
            duration=duration,
            num_clips=num_clips,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=mask_collator,
            num_workers=num_workers,
            world_size=0,
            rank=0,
            pin_mem=pin_mem,
            log_dir=None
        )
        self.train_loader = train_loader


        val_dataset, val_loader = init_data(
            data='videodataset',
            root_path=val_path,
            batch_size=batch_size,
            training=True,
            clip_len=num_frames,
            frame_sample_rate=sampling_rate,
            filter_short_videos=filter_short_videos,
            decode_one_clip=decode_one_clip,
            duration=duration,
            num_clips=num_clips,
            transform=transform,
            datasets_weights=datasets_weights,
            collator=mask_collator,
            num_workers=num_workers,
            world_size=0,
            rank=0,
            pin_mem=pin_mem,
            log_dir=None
        )
        self.val_loader = val_loader
        
        
        
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.decoder.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Loss functions
        self.recon_loss = nn.L1Loss()
        
        # Initialize wandb
        if config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project_name'],
                config=config
            )
    
    def _init_decoder(self):
        """Initialize decoder based on config"""
        model_cfg = self.config['model']
        import src.models.decoder as video_vitd

        decoder =  video_vitd.__dict__[model_cfg['model_name']](
            img_size=self.config['data']['crop_size'],
            patch_size=self.config['data']['patch_size'],
            num_frames=self.config['data']['num_frames'],
            tubelet_size=self.config['data']['tubelet_size'],
        ).to(self.device)
        return decoder

    def _load_checkpoint(self, checkpoint_path):
        """Load trained weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        msg = self.encoder.load_state_dict(checkpoint['encoder'])
        logger.info(f'loaded pretrained encoder with msg: {msg}')
        msg = self.predictor.load_state_dict(checkpoint['predictor'])
        logger.info(f'loaded pretrained predictor with msg: {msg}')
        # self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        # self.decoder.load_state_dict(checkpoint['decoder'])
    
    def load_clips(self, video, masks_enc, masks_pred):
        # -- unsupervised video clips
        # Put each clip on the GPU and concatenate along batch
        # dimension
        clips = torch.cat([u.to(self.device, non_blocking=True) for u in video[0]], dim=0)

        # Put each mask-enc/mask-pred pair on the GPU and reuse the
        # same mask pair for each clip
        _masks_enc, _masks_pred = [], []
        batch_size = self.config['data']['batch_size']
        num_clips = self.config['data']['num_clips']
        for _me, _mp in zip(masks_enc, masks_pred):
            _me = _me.to(self.device, non_blocking=True)
            _mp = _mp.to(self.device, non_blocking=True)
            _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
            _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
            _masks_enc.append(_me)
            _masks_pred.append(_mp)

        return (clips, _masks_enc, _masks_pred)
    
    def train_epoch(self, epoch):
        self.decoder.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (video, masks_enc, masks_pred) in enumerate(pbar):
                assert len(masks_enc) == len(masks_pred), 'Currently require num encoder masks = num predictor masks'

                clips, masks_enc, masks_pred = self.load_clips(video, masks_enc, masks_pred)
                B, C, T, H, W = clips.shape
                
                # Get embeddings from encoder
                with torch.no_grad():
                    # Get context embeddings using encoder mask
                    context_embeddings = self.encoder(clips, masks_enc)
                    
                    # Get predictions from predictor using both masks
                    predicted_embeddings = self.predictor(
                        ctxt=context_embeddings,
                        tgt=None,
                        masks_ctxt=masks_enc,
                        masks_tgt=masks_pred,
                        mask_index=0
                    )
                
                # Decode predictions to pixel space
                decoded_frames = self.decoder(predicted_embeddings)
                
                # Reshape clips and decoded frames for loss computation
                clips_reshaped = clips.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*T*H*W, C)
                decoded_reshaped = decoded_frames.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*T*H*W, C)
                
                # Get the target pixels using masks_pred
                target_pixels = torch.index_select(clips_reshaped, 0, masks_pred[0])
                predicted_pixels = torch.index_select(decoded_reshaped, 0, masks_pred[0])
                
                # Compute reconstruction loss only on masked regions
                loss = self.recon_loss(predicted_pixels, target_pixels)
                
                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # Log to wandb
                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'train_loss': loss.item(),
                        'train_batch': batch_idx + epoch * num_batches
                    })
                
                # Visualize occasionally
                if batch_idx % self.config['logging']['vis_interval'] == 0:
                    self._log_visualizations(
                        clips, decoded_frames, masks_pred,
                        f'train_vis_epoch_{epoch}_batch_{batch_idx}'
                    )
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, epoch):
        self.decoder.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        for batch_idx, (video, masks_enc, masks_pred) in enumerate(self.val_loader):
            # Move data to device
            clips, masks_enc, masks_pred = self.load_clips(video, masks_enc, masks_pred)
            B, C, T, H, W = clips.shape
            
            # Get embeddings from encoder
            with torch.no_grad():
                # Get context embeddings using encoder mask
                context_embeddings = self.encoder(clips, masks_enc)
                
                # Get predictions from predictor using both masks
                predicted_embeddings = self.predictor(
                    ctxt=context_embeddings,
                    tgt=None,
                    masks_ctxt=masks_enc,
                    masks_tgt=masks_pred,
                    mask_index=0
                )
            
            # Decode predictions to pixel space
            decoded_frames = self.decoder(predicted_embeddings)
            
            # Reshape clips and decoded frames for loss computation
            clips_reshaped = clips.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*T*H*W, C)
            decoded_reshaped = decoded_frames.permute(0, 2, 3, 4, 1).reshape(-1, C)  # (B*T*H*W, C)
            
            # Get the target pixels using masks_pred
            target_pixels = torch.index_select(clips_reshaped, 0, masks_pred[0])
            predicted_pixels = torch.index_select(decoded_reshaped, 0, masks_pred[0])
            
            # Compute reconstruction loss only on masked regions
            loss = self.recon_loss(predicted_pixels, target_pixels)
            total_loss += loss.item()
            
            # Visualize occasionally
            if batch_idx % self.config['logging']['vis_interval'] == 0:
                self._log_visualizations(
                    clips, decoded_frames, clips,
                    f'val_vis_epoch_{epoch}_batch_{batch_idx}'
                )
        
        val_loss = total_loss / num_batches
        if self.config['logging']['use_wandb']:
            wandb.log({
                'val_loss': val_loss,
                'val_epoch': epoch
            })
        
        return val_loss
    
    def _log_visualizations(self, video, predictions, masks_pred, name):
        """Create and log visualization grids showing masked input, predictions, and ground truth
    
        Args:
            video: Original video tensor (B,C,T,H,W)
            predictions: Predicted frames tensor (B,C,T,H,W) 
            masks_pred: Predictor masks indicating masked regions
            name: Name for logging
        """
        if not self.config['logging']['use_wandb']:
            return
            
        # Convert to numpy and move channels last
        def prep_for_vis(tensor):
            return tensor.cpu().numpy().transpose(0, 2, 3, 4, 1)  # (B,T,H,W,C)
        
        # Prepare tensors
        video_np = prep_for_vis(video)
        pred_np = prep_for_vis(predictions)
        
        # Create masked version by zeroing out masked regions
        masked_video = video.clone()
        B, C, T, H, W = video.shape
        flat_size = T * H * W
        
        # Create binary mask of same shape as video
        binary_mask = torch.ones((B, 1, flat_size), device=video.device)
        binary_mask.scatter_(2, masks_pred[0].unsqueeze(0).unsqueeze(0), 0)
        binary_mask = binary_mask.view(B, 1, T, H, W)
        binary_mask = binary_mask.expand(-1, C, -1, -1, -1)
        
        # Apply mask
        masked_video = masked_video * binary_mask
        masked_np = prep_for_vis(masked_video)
        
        # Take first item from batch and first frame
        # Stack horizontally: [masked input | prediction | ground truth]
        grid = np.concatenate([
            masked_np[0, 0],  # First frame of masked input
            pred_np[0, 0],    # First frame of prediction
            video_np[0, 0]    # First frame of ground truth
        ], axis=1)
        
        # Log to wandb
        wandb.log({
            name: wandb.Image(grid, caption="Masked Input | Prediction | Ground Truth")
        })
    
    def train(self, num_epochs):
        """Full training loop"""
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            print(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
                    'epoch': epoch,
                    'decoder_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, f'{self.config["checkpoints"]["path"]}/best_decoder.pth')


if __name__ == "__main__":
    # Load configs
    with open('configs/decoder_training.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Initialize trainer
    trainer = VJEPADecoderTrainer(
        config=config,
        device='cuda'
    )

    # Train
    trainer.train(num_epochs=config['training']['num_epochs'])