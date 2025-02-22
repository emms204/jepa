import torch
import torch.nn as nn

from src.models.utils.patch_embed import PatchEmbed, PatchEmbed3D
from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import trunc_normal_


class VisionTransformerDecoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=512,  # Usually smaller than encoder
        depth=8,  # Shallower than encoder
        num_heads=12,
        mlp_ratio=4.0,
        uniform_power=False,
        **kwargs
    ):
        """
        Initialize VisionTransformerDecoder.

        Args:
            img_size: The size of the input image.
            patch_size: The size of each patch in the input image.
            num_frames: The number of frames to use for the input video.
            tubelet_size: The tubelet size used for the input video.
            in_chans: The number of input channels.
            embed_dim: The number of dimensions in the embedding space.
            decoder_embed_dim: The embedding dimension of the decoder.
            depth: The number of transformer decoder layers.
            num_heads: The number of attention heads in each layer.
            mlp_ratio: The ratio of the number of channels in the MLP
                (hidden layer) to the embedding dimension.
            uniform_power: Whether to use uniform power normalization.
        """
        super().__init__()
        
        # Dimensions
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.uniform_power = uniform_power
        
        # Calculate number of patches
        if num_frames > 1:  # Video
            self.num_patches = (
                (num_frames // tubelet_size) *
                (img_size // patch_size) ** 2
            )
        else:  # Image
            self.num_patches = (img_size // patch_size) ** 2
            
        # Initial projection from encoder embedding dim to decoder embedding dim
        self.embed_proj = nn.Sequential(
            nn.Linear(embed_dim, decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, decoder_embed_dim)
        )
        
        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim),
            requires_grad=True
        )
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=decoder_embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=nn.LayerNorm
            ) for _ in range(depth)
        ])
        
        # Final normalization
        self.norm = nn.LayerNorm(decoder_embed_dim)
        
        # Nonlinear projection to patch space
        patch_dim = patch_size * patch_size * tubelet_size * in_chans
        self.patch_proj = nn.Sequential(
            nn.Linear(decoder_embed_dim, decoder_embed_dim * 2),
            nn.GELU(),
            nn.Linear(decoder_embed_dim * 2, decoder_embed_dim),
            nn.GELU(),
            nn.Linear(decoder_embed_dim, patch_dim),
        )
        
        # Optional: final activation for pixel values
        self.final_act = nn.Tanh()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        # Initialize positional embeddings
        """
        Initialize weights of the decoder model.

        This function initializes the positional embeddings and
        performs weight initialization for other layers
        """
        if self.num_frames > 1:
            # 3D positional embeddings for video
            pos_embed = get_3d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.img_size // self.patch_size,
                self.num_frames // self.tubelet_size,
                cls_token=False,
                uniform_power=self.uniform_power
            )
        else:
            # 2D positional embeddings for images
            pos_embed = get_2d_sincos_pos_embed(
                self.pos_embed.shape[-1],
                self.img_size // self.patch_size,
                cls_token=False
            )
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # Initialize other weights
        def _init_layer(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
                
        self.apply(_init_layer)

    def forward(self, x, return_intermediates=False):
        # x shape: (B, N, encoder_embed_dim)
        """
        Perform a forward pass through the VisionTransformerDecoder.

        Args:
            x (torch.Tensor): Input tensor from the encoder with shape
                (B, N, encoder_embed_dim) where B is the batch size, N is the
                number of patches, and encoder_embed_dim is the dimension of the
                encoder embeddings.
            return_intermediates (bool, optional): If True, returns the intermediate
                outputs from each transformer block in addition to the final output.

        Returns:
            torch.Tensor or tuple: If `return_intermediates` is False, returns the
                output tensor reshaped to the original image or video dimensions.
                If True, returns a tuple containing the output tensor and a list
                of intermediate outputs from each transformer block.
        """

        B = x.shape[0]
        
        # Project to decoder embedding dimension
        x = self.embed_proj(x)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Store intermediates if requested
        intermediates = []
        
        # Process through transformer blocks
        for blk in self.blocks:
            x = blk(x)
            if return_intermediates:
                intermediates.append(x)
        
        x = self.norm(x)
        
        # Project to patch dimension
        x = self.patch_proj(x)  # (B, N, patch_dim)
        
        # Reshape to video/image dimensions
        H = W = self.img_size // self.patch_size
        if self.num_frames > 1:
            # Video reshaping
            T = self.num_frames // self.tubelet_size
            x = x.reshape(B, T, H, W, 
                         self.tubelet_size, self.patch_size, self.patch_size, 
                         self.in_chans)
            x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
            x = x.reshape(B, self.in_chans, 
                         self.num_frames, 
                         self.img_size, 
                         self.img_size)
        else:
            # Image reshaping
            x = x.reshape(B, H, W, 
                         self.patch_size, self.patch_size, 
                         self.in_chans)
            x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
            x = x.reshape(B, self.in_chans, 
                         self.img_size, 
                         self.img_size)
        
        # Apply final activation
        x = self.final_act(x)
        
        if return_intermediates:
            return x, intermediates
        return x

def vit_tiny(patch_size=16, **kwargs):
    return VisionTransformerDecoder(
        img_size=224,
        patch_size=patch_size,
        num_frames=1,
        tubelet_size=2,
        embed_dim=192,
        decoder_embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4.0,
        uniform_power=False,
        **kwargs
    )       

def vit_small(patch_size=16, **kwargs):
    return VisionTransformerDecoder(
        img_size=224,
        patch_size=patch_size,
        num_frames=1,
        tubelet_size=2,
        embed_dim=384,
        decoder_embed_dim=384,
        depth=12,
        num_heads=6,
        mlp_ratio=4.0,
        uniform_power=False,
        **kwargs
    )

def vit_base(patch_size=16, **kwargs):
    return VisionTransformerDecoder(
        img_size=224,
        patch_size=patch_size,
        num_frames=1,
        tubelet_size=2,
        embed_dim=768,
        decoder_embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        uniform_power=False,
        **kwargs
    )

def vit_large(patch_size=16, **kwargs):
    return VisionTransformerDecoder(
        img_size=224,
        patch_size=patch_size,
        num_frames=1,
        tubelet_size=2,
        embed_dim=1024,
        decoder_embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        uniform_power=False,
        **kwargs
    )

def vit_huge(patch_size=16, **kwargs):
    return VisionTransformerDecoder(
        img_size=224,
        patch_size=patch_size,
        num_frames=1,
        tubelet_size=2,
        embed_dim=1280,
        decoder_embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4.0, 
        uniform_power=False,
        **kwargs
    )