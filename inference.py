import torch
import torch.nn.functional as F
from pathlib import Path
import yaml
import logging
import decord
import numpy as np
from src.utils.tensors import repeat_interleave_batch
from src.datasets.utils.video.transforms import VideoTransform

class VJEPAInference:
    def __init__(self, config_path, checkpoint_path):
        # Load config
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.encoder = self._init_encoder()  # X-encoder
        self.predictor = self._init_predictor()
        self.target_encoder = self._init_encoder()  # Y-encoder
        
        # Load checkpoint
        self._load_checkpoint(checkpoint_path)
        
        # Set models to eval mode
        self.encoder.eval()
        self.predictor.eval()
        self.target_encoder.eval()
        
        # Setup video transform
        self.transform = VideoTransform(
            training=False,
            crop_size=self.config['data']['crop_size'],
        )

    def _init_encoder(self):
        # Import your specific encoder architecture
        from src.models.vision_transformer import VisionTransformer
        
        model_cfg = self.config['model']
        encoder = VisionTransformer(
            img_size=self.config['data']['crop_size'],
            patch_size=self.config['data']['patch_size'],
            tubelet_size=self.config['data']['tubelet_size'],
            in_chans=3,
            embed_dim=model_cfg['embed_dim'],
            depth=model_cfg['depth'],
            num_heads=model_cfg['num_heads'],
            mlp_ratio=model_cfg['mlp_ratio'],
            use_act_checkpoint=False,
            use_abs_pos=True
        )
        return encoder

    def _init_predictor(self, encoder):
        """Initialize predictor based on config"""
        model_cfg = self.config['model']
        # Initialize predictor using vit_predictor from src.models.predictor
        from src.models.predictor import vit_predictor
        
        predictor = vit_predictor(
            img_size=self.config['data']['crop_size'],
            patch_size=self.config['data']['patch_size'], 
            num_frames=self.config['data']['num_frames'],
            tubelet_size=self.config['data']['tubelet_size'],
            embed_dim=model_cfg['embed_dim'],
            predictor_embed_dim=model_cfg['pred_embed_dim'],
            depth=model_cfg['pred_depth'],
            num_heads=model_cfg['num_heads'],
            uniform_power=model_cfg.get('uniform_power', True),
            use_mask_tokens=model_cfg.get('use_mask_tokens', True),
            num_mask_tokens=model_cfg.get('num_mask_tokens', 2),
            zero_init_mask_tokens=model_cfg.get('zero_init_mask_tokens', True),
            use_sdpa=self.config['meta'].get('use_sdpa', False)
        )
        # Import and initialize your predictor model here based on model_cfg
        return predictor.to(self.device)
    
    def _init_decoder(self):
        """Initialize decoder based on config"""
        model_cfg = self.config['model']
        from src.models.decoder import VisionTransformerDecoder 

        decoder = VisionTransformerDecoder(
            img_size=self.config['data']['crop_size'],
            patch_size=self.config['data']['patch_size'],
            num_frames=self.config['data']['num_frames'],
            tubelet_size=self.config['data']['tubelet_size'],
            in_chans=3, # Or self.config['data']['in_chans'] if you have it in config
            embed_dim=model_cfg['pred_embed_dim'], # Decoder input dim should match predictor output dim
            depth=model_cfg['pred_depth'], # You can use same depth as predictor or configure separately
            num_heads=model_cfg['num_heads'], # You can use same num_heads or configure separately
            mlp_ratio=model_cfg['mlp_ratio'],
        ).to(self.device)
        return decoder

    def _load_checkpoint(self, checkpoint_path):
        """Load trained weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.predictor.load_state_dict(checkpoint['predictor'])
        # self.target_encoder.load_state_dict(checkpoint['target_encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

    def _load_video(self, video_path):
        """Load and preprocess video"""
        vr = decord.VideoReader(str(video_path))
        
        # Get frames based on config
        frames_per_clip = self.config['data']['num_frames']
        frame_step = self.config['data']['sampling_rate']
        frame_indices = list(range(0, len(vr), frame_step))[:frames_per_clip]
        
        # Read frames
        frames = vr.get_batch(frame_indices).asnumpy()
        
        # Apply transforms
        frames = self.transform(frames)
        return frames.unsqueeze(0).to(self.device)  # Add batch dimension

    def _apply_mask(self, video, mask_cfg):
        B, C, T, H, W = video.shape
        
        # Calculate patch dimensions
        pH = H // self.config['data']['patch_size']
        pW = W // self.config['data']['patch_size']
        pT = T // self.config['data']['tubelet_size']
        
        # Generate mask based on mask_cfg
        num_patches = pH * pW * pT
        num_mask = int(num_patches * mask_cfg['mask_ratio'])
        
        # Randomly select patches to mask
        noise = torch.rand(B, num_patches, device=video.device)
        mask_indices = torch.argsort(noise, dim=1)[:, :num_mask]
        unmasked_indices = torch.argsort(noise, dim=1)[:, num_mask:]
        
        # Create binary mask
        mask = torch.ones((B, num_patches), device=video.device)
        mask.scatter_(1, mask_indices, 0)
        
        # Apply mask to video
        mask_3d = mask.reshape(B, pT, pH, pW).unsqueeze(1).repeat(1, C, 1, 1, 1)
        mask_3d = F.interpolate(mask_3d, size=(T, H, W), mode='nearest')
        masked_video = video * mask_3d
        
        return masked_video, mask, unmasked_indices, mask_indices

    @torch.no_grad()
    def predict(self, video_path):
        """Perform inference on a video"""
        video = self._load_video(video_path)
        
        # Apply masking
        masked_video, mask, unmasked_indices, masked_indices = self._apply_mask(
            video, self.config['mask'][0]
        )
        
        # Get context embeddings
        context_embeddings = self.encoder(masked_video, unmasked_indices)
        
        # For inference, we use mask tokens (not target tokens)
        # Set tgt=None since we don't have target embeddings during inference
        predictions = self.predictor(
            ctxt=context_embeddings,
            tgt=None,  # No target tokens during inference
            masks_ctxt=unmasked_indices,
            masks_tgt=masked_indices,
            mask_index=0  # Use first mask token
        )

        pixel_predictions = self.decoder(predictions)
        
        return {
            'predictions': pixel_predictions,
            'mask': mask,
            'masked_video': masked_video
        }

    def visualize_predictions(self, results):
        import matplotlib.pyplot as plt
        
        # Get results
        masked_video = results['masked_video']
        predictions = results['pixel_predictions']
        mask = results['mask']
        
        # Convert tensors to numpy arrays
        masked_video = masked_video.cpu().numpy()
        predictions = predictions.cpu().numpy()
        
        # Plot results
        fig, axes = plt.subplots(2, 1, figsize=(10, 10))
        
        # Plot masked video
        axes[0].imshow(masked_video[0].transpose(1, 2, 0))
        axes[0].set_title('Masked Input')
        
        # Plot predictions
        axes[1].imshow(predictions[0].transpose(1, 2, 0))
        axes[1].set_title('Predictions')
        
        plt.tight_layout()
        plt.show()

# Example usage
if __name__ == "__main__":
    config_path = "configs/pretrain/vitl16.yaml"
    checkpoint_path = "path/to/your/checkpoint.pth"
    video_path = "path/to/test/video.mp4"
    
    # Initialize inference
    jepa = VJEPAInference(config_path, checkpoint_path)
    
    # Run inference
    results = jepa.predict(video_path)
    
    # Visualize results
    jepa.visualize_predictions(results)


