import torch
import os
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import wandb
import numpy as np
from tqdm import tqdm
import yaml
import logging

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
        logger.info("Initializing VJEPADecoderTrainer")
        logger.info(f"Using device: {device}")
        
        # Setup device
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # -- META
        cfgs_meta = self.config.get('meta')
        pretrain_folder = cfgs_meta.get('folder', None)
        ckp_fname = cfgs_meta.get('checkpoint', None)
        use_sdpa = cfgs_meta.get('use_sdpa', False)
        which_dtype = cfgs_meta.get('dtype')
        
        logger.info(f"Loading checkpoint from: {os.path.join(pretrain_folder, ckp_fname)}")
        logger.info(f"Using SDPA: {use_sdpa}, dtype: {which_dtype}")
        
        if which_dtype.lower() == 'bfloat16':
            dtype = torch.bfloat16
            mixed_precision = True
        elif which_dtype.lower() == 'float16':
            dtype = torch.float16
            mixed_precision = True
        else:
            dtype = torch.float32
            mixed_precision = False
        
        self.dtype = dtype
        self.mixed_precision = mixed_precision
        logger.info(f"Using dtype: {dtype}, mixed precision: {mixed_precision}")

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

        # -- LOSS
        cfgs_loss = self.config.get('loss')
        self.loss_exp = cfgs_loss.get('loss_exp')
        self.reg_coeff = cfgs_loss.get('reg_coeff')

        # -- OPTIMIZATION
        cfgs_opt = self.config.get('optimization')
        ipe = cfgs_opt.get('ipe', None)
        ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
        clip_grad = cfgs_opt.get('clip_grad', None)
        wd = float(cfgs_opt.get('weight_decay'))
        final_wd = float(cfgs_opt.get('final_weight_decay'))
        num_epochs = cfgs_opt.get('epochs')
        warmup = cfgs_opt.get('warmup')
        start_lr = cfgs_opt.get('start_lr')
        lr = cfgs_opt.get('lr')
        final_lr = cfgs_opt.get('final_lr')
        ema = cfgs_opt.get('ema')
        betas = cfgs_opt.get('betas', (0.9, 0.999))
        eps = cfgs_opt.get('eps', 1.e-8)

        # Initialize models
        logger.info("Initializing encoder and predictor models")
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
        
        logger.info("Loading pretrained weights")
        self._load_checkpoint(pretrained_path)
        
        logger.info("Initializing decoder")
        self.decoder = self._init_decoder()
        
        # Freeze encoder and predictor
        logger.info("Freezing encoder and predictor parameters")
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
        
        # Initialize optimizer and loss
        logger.info("Initializing optimizer and loss functions")
        self.optimizer = optim.AdamW(
            self.decoder.parameters(),
            lr=self.config['training']['lr'],
            weight_decay=self.config['training']['weight_decay'],
            betas=betas,
            eps=eps
        )
        self.scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        self.recon_loss = nn.L1Loss()
        
        # Initialize wandb
        if config['logging']['use_wandb']:
            logger.info("Initializing wandb")
            wandb.init(
                project=config['logging']['project_name'],
                config=config
            )
    
    def _init_decoder(self):
        """Initialize decoder based on config"""
        model_cfg = self.config['model']
        import src.models.decoder as video_vitd

        # Get predictor's output dimension
        predictor_out_dim = self.predictor.backbone.predictor_proj.out_features

        # Create kwargs with matching configuration
        decoder_kwargs = {
            'img_size': self.config['data']['crop_size'],
            'patch_size': self.config['data']['patch_size'],
            'num_frames': self.config['data']['num_frames'],
            'tubelet_size': self.config['data']['tubelet_size'],
            'in_chans': 3,
            'embed_dim': predictor_out_dim,  # Input dimension matches predictor output
            'decoder_embed_dim': self.config['model'].get('decoder_embed_dim', predictor_out_dim),  # Usually smaller
            'depth': self.config['model'].get('decoder_depth', 12),
            'num_heads': self.config['model'].get('decoder_heads', 12),
        }

        decoder = video_vitd.__dict__[model_cfg['decoder_type']](**decoder_kwargs)
        return decoder.to(self.device)

    def _load_checkpoint(self, checkpoint_path):
        """Load trained weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        pretrained_dict = checkpoint['encoder']
        
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
        for k, v in self.encoder.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                pretrained_dict[k] = v
        msg = self.encoder.load_state_dict(pretrained_dict, strict=False)
        print(self.encoder)
        # logger.info(f'loaded pretrained model with msg: {msg}')
        logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {checkpoint_path}')
        
        pretrained_dict = checkpoint['predictor']
        pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
        pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
        for k, v in self.predictor.state_dict().items():
            if k not in pretrained_dict:
                logger.info(f'key "{k}" could not be found in loaded state dict')
            elif pretrained_dict[k].shape != v.shape:
                logger.info(f'key "{k}" is of different shape in model and loaded state dict')
                pretrained_dict[k] = v
        msg = self.predictor.load_state_dict(pretrained_dict, strict=False)
        print(self.predictor)
        # logger.info(f'loaded pretrained model with msg: {msg}')
        logger.info(f'loaded pretrained predictor from epoch: {checkpoint["epoch"]}\n path: {checkpoint_path}')
        
        # self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        # self.decoder.load_state_dict(checkpoint['decoder'])
    
    def load_clips(self, video, masks_enc, masks_pred):
        logger.debug("Loading clips and masks to device")
        logger.debug(f"Input video shape: {[v.shape for v in video[0]]}")
        
        clips = torch.cat([u.to(self.device, non_blocking=True) for u in video[0]], dim=0)
        logger.debug(f"Concatenated clips shape: {clips.shape}")
        
        _masks_enc, _masks_pred = [], []
        batch_size = self.config['data']['batch_size']
        num_clips = self.config['data']['num_clips']
        
        for i, (_me, _mp) in enumerate(zip(masks_enc, masks_pred)):
            _me = _me.to(self.device, non_blocking=True)
            _mp = _mp.to(self.device, non_blocking=True)
            logger.debug(f"Mask pair {i+1} shapes - Encoder: {_me.shape}, Predictor: {_mp.shape}")
            
            _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
            _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
            logger.debug(f"After repeat - Encoder: {_me.shape}, Predictor: {_mp.shape}")
            
            _masks_enc.append(_me)
            _masks_pred.append(_mp)
        
        return clips, _masks_enc, _masks_pred
    
    def train_epoch(self, epoch):
        self.decoder.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        logger.info(f"Starting training epoch {epoch}")
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (video, masks_enc, masks_pred) in enumerate(pbar):
                logger.debug(f"Processing batch {batch_idx+1}/{num_batches}")
                
                clips, masks_enc, masks_pred = self.load_clips(video, masks_enc, masks_pred)
                B, C, T, H, W = clips.shape
                logger.debug(f"Processed clips shape: {clips.shape}")
                
                # Get embeddings from encoder
                with torch.no_grad():
                    logger.debug("Computing encoder embeddings")
                    context_embeddings = self.encoder(clips, masks_enc)
                    logger.debug(f"Context embeddings shape: {context_embeddings.shape}")
                    
                    logger.debug("Computing predictor embeddings")
                    predicted_embeddings = self.predictor(
                        ctxt=context_embeddings,
                        tgt=None,
                        masks_ctxt=masks_enc,
                        masks_tgt=masks_pred,
                        mask_index=0
                    )
                    logger.debug(f"Predicted embeddings shape: {predicted_embeddings.shape}")
                
                # Decode predictions
                with torch.cuda.amp.autocast(dtype=self.dtype, enabled=self.mixed_precision):
                    logger.debug("Decoding predictions")
                    decoded_frames_list = self.decoder(
                        ctxts=context_embeddings,
                        tgts=predicted_embeddings,
                        masks_ctxts=masks_enc,
                        masks_tgts=masks_pred
                    )
                    logger.debug(f"Decoded frames shapes: {[df.shape for df in decoded_frames_list]}")
                    
                    loss_jepa = 0
                    loss_reg = 0
                    
                    for i, (decoded_frame, mask_pred) in enumerate(zip(decoded_frames_list, masks_pred)):
                        mask_indices = mask_pred.flatten().nonzero().squeeze()
                        logger.debug(f"Mask {i+1} - Number of masked indices: {len(mask_indices)}")
                        
                        target_pixels = clips.permute(0, 2, 3, 4, 1).reshape(-1, C)[mask_indices]
                        pred_pixels = decoded_frame.permute(0, 2, 3, 4, 1).reshape(-1, C)[mask_indices]
                        logger.debug(f"Target/Pred pixels shape: {target_pixels.shape}")
                        
                        # Compute losses
                        curr_loss_jepa = torch.mean(torch.abs(pred_pixels - target_pixels)**self.loss_exp) / self.loss_exp
                        pstd = torch.sqrt(torch.var(pred_pixels, dim=1) + 1e-6)
                        curr_loss_reg = torch.mean(F.relu(1. - pstd))
                        
                        loss_jepa += curr_loss_jepa
                        loss_reg += curr_loss_reg
                        
                        logger.debug(f"Mask {i+1} - JEPA loss: {curr_loss_jepa:.4f}, Reg loss: {curr_loss_reg:.4f}")
                    
                    loss_jepa /= len(masks_pred)
                    loss = loss_jepa + self.reg_coeff * loss_reg
                    
                    logger.debug(f"Final losses - JEPA: {loss_jepa:.4f}, Reg: {loss_reg:.4f}, Total: {loss:.4f}")
                
                # Backprop
                if self.mixed_precision:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                else:
                    loss.backward()
                
                if self.config['optimization']['clip_grad'] is not None:
                    _dec_norm = torch.nn.utils.clip_grad_norm_(
                        self.decoder.parameters(), 
                        self.config['optimization']['clip_grad']
                    )
                    logger.debug(f"Gradient norm after clipping: {_dec_norm}")
                
                if self.mixed_precision:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Update metrics
                total_loss += loss.item()
                pbar.set_postfix({'loss': loss.item()})
                
                # Log to wandb
                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'train_loss': loss.item(),
                        'train_loss_jepa': loss_jepa.item(),
                        'train_loss_reg': loss_reg.item(),
                        'train_batch': batch_idx + epoch * num_batches,
                        'grad_norm': _dec_norm if self.config['optimization']['clip_grad'] is not None else None
                    })
                
                # Visualize occasionally
                if batch_idx % self.config['logging']['vis_interval'] == 0:
                    logger.info(f"Creating visualizations for batch {batch_idx}")
                    self._log_visualizations(
                        clips,
                        decoded_frames_list,
                        masks_pred,
                        f'train_vis_epoch_{epoch}_batch_{batch_idx}'
                    )
        
        avg_loss = total_loss / num_batches
        logger.info(f"Epoch {epoch} completed. Average loss: {avg_loss:.4f}")
        return avg_loss
    
    @torch.no_grad()
    def validate(self, epoch):
        self.decoder.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        logger.info(f"Starting validation epoch {epoch}")
        
        with tqdm(self.val_loader, desc=f'Validation Epoch {epoch}') as pbar:
            for batch_idx, (video, masks_enc, masks_pred) in enumerate(pbar):
                logger.debug(f"Processing validation batch {batch_idx+1}/{num_batches}")
                
                clips, masks_enc, masks_pred = self.load_clips(video, masks_enc, masks_pred)
                B, C, T, H, W = clips.shape
                
                # Get embeddings
                context_embeddings = self.encoder(clips, masks_enc)
                predicted_embeddings = self.predictor(
                    ctxt=context_embeddings,
                    tgt=None,
                    masks_ctxt=masks_enc,
                    masks_tgt=masks_pred,
                    mask_index=0
                )
                
                # Decode predictions
                with torch.cuda.amp.autocast(dtype=self.dtype, enabled=self.mixed_precision):
                    decoded_frames_list = self.decoder(
                        ctxts=context_embeddings,
                        tgts=predicted_embeddings,
                        masks_ctxts=masks_enc,
                        masks_tgts=masks_pred
                    )
                    
                    loss_jepa = 0
                    loss_reg = 0
                    
                    for decoded_frame, mask_pred in zip(decoded_frames_list, masks_pred):
                        mask_indices = mask_pred.flatten().nonzero().squeeze()
                        target_pixels = clips.permute(0, 2, 3, 4, 1).reshape(-1, C)[mask_indices]
                        pred_pixels = decoded_frame.permute(0, 2, 3, 4, 1).reshape(-1, C)[mask_indices]
                        
                        loss_jepa += torch.mean(torch.abs(pred_pixels - target_pixels)**self.loss_exp) / self.loss_exp
                        pstd = torch.sqrt(torch.var(pred_pixels, dim=1) + 1e-6)
                        loss_reg += torch.mean(F.relu(1. - pstd))
                    
                    loss_jepa /= len(masks_pred)
                    loss = loss_jepa + self.reg_coeff * loss_reg
                    
                    logger.debug(f"Validation batch {batch_idx} - Loss: {loss:.4f}")
                
                total_loss += loss.item()
                
                # Visualize occasionally
                if batch_idx % self.config['logging']['vis_interval'] == 0:
                    logger.info(f"Creating validation visualizations for batch {batch_idx}")
                    self._log_visualizations(
                        clips,
                        decoded_frames_list,
                        masks_pred,
                        f'val_vis_epoch_{epoch}_batch_{batch_idx}'
                    )
        
        val_loss = total_loss / num_batches
        logger.info(f"Validation epoch {epoch} completed. Average loss: {val_loss:.4f}")
        
        if self.config['logging']['use_wandb']:
            wandb.log({
                'val_loss': val_loss,
                'val_epoch': epoch
            })
        
        return val_loss
    
    def _log_visualizations(self, video, decoded_frames_list, masks_pred_list, name):
        """Create and log visualization grids showing context, predictions, and ground truth for multiple masks
    
        Args:
            video: Original video tensor (B,C,T,H,W)
            decoded_frames_list: List of predicted frames tensors for each mask
            masks_pred_list: List of predictor masks indicating masked regions
            name: Name for logging
        """
        if not self.config['logging']['use_wandb']:
            return
            
        # Convert to numpy and move channels last
        def prep_for_vis(tensor):
            return tensor.cpu().detach().numpy().transpose(0, 2, 3, 4, 1)  # (B,T,H,W,C)
        
        # Select random subset of masks to visualize (2-3 masks)
        num_masks = min(3, len(masks_pred_list))
        mask_indices = torch.randperm(len(masks_pred_list))[:num_masks]
        
        video_np = prep_for_vis(video)
        
        for idx, mask_idx in enumerate(mask_indices):
            mask_pred = masks_pred_list[mask_idx]
            decoded_frame = decoded_frames_list[mask_idx]
            pred_np = prep_for_vis(decoded_frame)
            
            # Create context version (original video with masked regions zeroed)
            context_video = video.clone()
            B, C, T, H, W = video.shape
            flat_size = T * H * W
            
            # Create binary mask of same shape as video
            binary_mask = torch.ones((B, 1, flat_size), device=video.device)
            binary_mask.scatter_(2, mask_pred.unsqueeze(0).unsqueeze(0), 0)
            binary_mask = binary_mask.view(B, 1, T, H, W)
            binary_mask = binary_mask.expand(-1, C, -1, -1, -1)
            
            # Apply mask to get context
            context_video = context_video * binary_mask
            context_np = prep_for_vis(context_video)
            
            # Take first item from batch and first frame
            # Stack horizontally: [context | prediction | ground truth]
            grid = np.concatenate([
                context_np[0, 0],    # First frame of context (masked input)
                pred_np[0, 0],       # First frame of prediction
                video_np[0, 0]       # First frame of ground truth
            ], axis=1)
            
            # Log to wandb with mask index
            wandb.log({
                f"{name}_mask_{idx}": wandb.Image(
                    grid, 
                    caption=f"Mask {idx}: Context | Prediction | Ground Truth"
                )
            })
            
            # Optionally log temporal visualization (e.g., first 4 frames)
            if T > 1:
                temporal_grid = np.concatenate([
                    np.concatenate([context_np[0, t] for t in range(min(4, T))], axis=1),
                    np.concatenate([pred_np[0, t] for t in range(min(4, T))], axis=1),
                    np.concatenate([video_np[0, t] for t in range(min(4, T))], axis=1)
                ], axis=0)
                
                wandb.log({
                    f"{name}_temporal_mask_{idx}": wandb.Image(
                        temporal_grid,
                        caption=f"Mask {idx} Temporal: Context | Prediction | Ground Truth"
                    )
                })
    
    def train(self, num_epochs):
        logger.info(f"Starting training for {num_epochs} epochs")
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            
            logger.info(f'Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = f'{self.config["checkpoints"]["path"]}/best_decoder.pth'
                logger.info(f"New best validation loss: {val_loss:.4f}. Saving model to {save_path}")
                
                torch.save({
                    'epoch': epoch,
                    'decoder_state_dict': self.decoder.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)


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