import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import wandb
import numpy as np
from tqdm import tqdm

class VJEPADecoderTrainer:
    def __init__(
        self,
        encoder,
        predictor,
        decoder,
        train_loader,
        val_loader,
        config,
        device='cuda'
    ):
        # Models
        self.encoder = encoder.to(device)
        self.predictor = predictor.to(device)
        self.decoder = decoder.to(device)
        
        # Freeze encoder and predictor
        for model in [self.encoder, self.predictor]:
            for param in model.parameters():
                param.requires_grad = False
        
        # Data
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Training config
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.decoder.parameters(),
            lr=config['training']['lr'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Loss functions
        self.recon_loss = nn.L1Loss()
        
        # Initialize wandb
        if config['logging']['use_wandb']:
            wandb.init(
                project=config['logging']['project_name'],
                config=config
            )
    
    def train_epoch(self, epoch):
        self.decoder.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f'Epoch {epoch}') as pbar:
            for batch_idx, (video, _) in enumerate(pbar):
                video = video.to(self.device)
                B = video.shape[0]
                
                # Apply masking
                masked_video, mask, unmasked_indices, masked_indices = self._apply_mask(
                    video, self.config['mask'][0]
                )
                
                # Get embeddings from encoder
                with torch.no_grad():
                    context_embeddings = self.encoder(masked_video, unmasked_indices)
                    
                    # Get predictions from predictor
                    predicted_embeddings = self.predictor(
                        ctxt=context_embeddings,
                        tgt=None,
                        masks_ctxt=unmasked_indices,
                        masks_tgt=masked_indices,
                        mask_index=0
                    )
                
                # Decode predictions to pixel space
                decoded_frames = self.decoder(predicted_embeddings)
                
                # Compute reconstruction loss
                loss = self.recon_loss(decoded_frames, video)
                
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
                        masked_video, decoded_frames, video,
                        f'train_vis_epoch_{epoch}_batch_{batch_idx}'
                    )
        
        return total_loss / num_batches
    
    @torch.no_grad()
    def validate(self, epoch):
        self.decoder.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        for batch_idx, (video, _) in enumerate(self.val_loader):
            video = video.to(self.device)
            
            # Same forward pass as training
            masked_video, mask, unmasked_indices, masked_indices = self._apply_mask(
                video, self.config['mask'][0]
            )
            
            context_embeddings = self.encoder(masked_video, unmasked_indices)
            predicted_embeddings = self.predictor(
                ctxt=context_embeddings,
                tgt=None,
                masks_ctxt=unmasked_indices,
                masks_tgt=masked_indices,
                mask_index=0
            )
            
            decoded_frames = self.decoder(predicted_embeddings)
            loss = self.recon_loss(decoded_frames, video)
            
            total_loss += loss.item()
            
            # Visualize occasionally
            if batch_idx % self.config['logging']['vis_interval'] == 0:
                self._log_visualizations(
                    masked_video, decoded_frames, video,
                    f'val_vis_epoch_{epoch}_batch_{batch_idx}'
                )
        
        val_loss = total_loss / num_batches
        if self.config['logging']['use_wandb']:
            wandb.log({
                'val_loss': val_loss,
                'val_epoch': epoch
            })
        
        return val_loss
    
    def _log_visualizations(self, masked_video, predictions, targets, name):
        """Create and log visualization grids"""
        if not self.config['logging']['use_wandb']:
            return
            
        # Convert to numpy and normalize
        def prep_for_vis(tensor):
            return tensor.cpu().numpy().transpose(0, 2, 3, 1)
        
        masked_frames = prep_for_vis(masked_video)
        pred_frames = prep_for_vis(predictions)
        target_frames = prep_for_vis(targets)
        
        # Create visualization grid
        grid = np.concatenate([
            masked_frames[0], pred_frames[0], target_frames[0]
        ], axis=1)
        
        wandb.log({
            name: wandb.Image(grid, caption="Masked | Predicted | Target")
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

    # Initialize models
    encoder = vit_encoder_tiny()
    predictor = vit_predictor_tiny()
    decoder = vit_decoder_tiny()

    # Initialize trainer
    trainer = VJEPADecoderTrainer(
        encoder=encoder,  # Your pretrained encoder
        predictor=predictor,  # Your pretrained predictor
        decoder=decoder,  # Your initialized decoder
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device='cuda'
    )

    # Train
    trainer.train(num_epochs=config['training']['num_epochs'])