import torch
import torch.nn as nn

from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import trunc_normal_
from functools import partial


class VisionTransformerDecoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        use_pos_embed=True,  # Added: Option to use positional embeddings
        pos_embed_type='sincos', # Added: Type of positional embedding ('sincos', 'learnable', None)
        nonlinear_proj_head=False, # Added: Option for nonlinear projection head
        projection_layers=2, # Added: Number of layers in projection head if nonlinear_proj_head is True
        intermediate_dim_proj=2048, # Added: Intermediate dimension for nonlinear projection head
        decoder_embed_dim=None, # Added: Option to have different embed_dim for decoder
        uniform_power=False, # Added: uniform_power for positional embeddings
        **kwargs
    ):
        super().__init__()

        self.embed_dim = decoder_embed_dim if decoder_embed_dim is not None else embed_dim # Use decoder_embed_dim if provided, else use encoder embed_dim
        self.patch_size = patch_size
        self.tubelet_size = tubelet_size
        self.img_size = img_size
        self.num_frames = num_frames
        self.in_chans = in_chans
        self.is_video = num_frames > 1
        self.use_pos_embed = use_pos_embed
        self.pos_embed_type = pos_embed_type
        self.uniform_power = uniform_power


        # 1. Positional Embedding (Optional)
        self.pos_embed = None
        if use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, self._get_num_patches(), self.embed_dim),
                requires_grad=(pos_embed_type == 'learnable')) # Learnable if type is 'learnable'
            self._init_pos_embed(self.pos_embed.data) # Initialize if using sincos

        # 2. Transformer Decoder Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=self.embed_dim, # Use decoder embed_dim
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                norm_layer=norm_layer
            ) for _ in range(depth)]) # Depth is still configurable via 'depth' parameter

        self.norm = norm_layer(self.embed_dim) # Use decoder embed_dim


        # 3. Projection Head (Linear or Nonlinear)
        patch_dim = patch_size * patch_size * tubelet_size * in_chans
        if nonlinear_proj_head:
            proj_layers = []
            current_dim = self.embed_dim # Use decoder embed_dim
            for _ in range(projection_layers - 1): # Stacked linear layers with GELU
                proj_layers.extend([
                    nn.Linear(current_dim, intermediate_dim_proj),
                    nn.GELU()
                ])
                current_dim = intermediate_dim_proj
            proj_layers.append(nn.Linear(current_dim, patch_dim)) # Final projection to patch_dim
            self.embed_to_patches = nn.Sequential(*proj_layers)
        else:
            self.embed_to_patches = nn.Linear(self.embed_dim, patch_dim) # Linear projection


        # 4. Patch to Video using ConvTranspose3d (Upsampling and Channel Projection)
        self.patch_to_video = nn.ConvTranspose3d(
            self.embed_dim,  # Input channels from the last Transformer block (decoder_embed_dim)
            in_chans,
            kernel_size=(tubelet_size, patch_size, patch_size),
            stride=(tubelet_size, patch_size, patch_size),
        )

        self.init_weights() # Initialize weights


    def init_weights(self):
        """Initialize decoder weights"""
        self.apply(self._init_weights)
        self._rescale_blocks()
        if self.pos_embed is not None and self.pos_embed_type == 'learnable':
             trunc_normal_(self.pos_embed, std=0.02) # Initialize learnable pos_embed


    def _get_num_patches(self):
        """Helper to calculate number of patches"""
        grid_size = self.img_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size
        if self.is_video:
            return grid_depth * grid_size * grid_size
        else:
            return grid_size * grid_size


    def _init_pos_embed(self, pos_embed):
        """Initialize positional embedding (sincos)"""
        embed_dim = pos_embed.size(-1)
        grid_size = self.img_size // self.patch_size
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.uniform_power
            )
        else:
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.ConvTranspose3d): # Initialize ConvTranspose3d weights
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


    def _rescale_blocks(self):
        """Rescale decoder blocks - similar to encoder/predictor"""
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)


    def forward(self, x):
        # x shape: (B, N, embed_dim)
        B, N, _ = x.shape


        # 1. Add Positional Embedding (if enabled)
        if self.pos_embed is not None:
            x = x + self.pos_embed.repeat(B, 1, 1)


        # 2. Transformer Decoder Blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)


        # 3. Projection to Patches (Linear or Nonlinear)
        x = self.embed_to_patches(x)  # (B, N, patch_dim)


        # 4. Reshape to Patch Grid
        H_patches = self.img_size // self.patch_size
        W_patches = self.img_size // self.patch_size
        T_patches = self.num_frames // self.tubelet_size

        x = x.reshape(B, T_patches, H_patches, W_patches, -1) # Infer patch_dim
        patch_dim = x.shape[-1] # Dynamically get patch_dim after reshape

        x = x.reshape(B, T_patches, H_patches, W_patches,
                      self.tubelet_size, self.patch_size, self.patch_size,
                      self.in_chans)


        # 5. Rearrange to final video shape
        x = x.permute(0, 7, 1, 4, 2, 5, 3, 6).contiguous()
        video = x.reshape(B, self.in_chans,
                          self.num_frames,
                          self.img_size,
                          self.img_size)


        return video

