import torch
import torch.nn as nn
import math
import logging

from src.models.utils.patch_embed import PatchEmbed, PatchEmbed3D
from src.models.utils.modules import Block
from src.models.utils.pos_embs import get_2d_sincos_pos_embed, get_3d_sincos_pos_embed
from src.utils.tensors import trunc_normal_
from src.masks.utils import apply_masks
from functools import partial

logger = logging.getLogger(__name__)

class VisionTransformerDecoder(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        num_frames=1,
        tubelet_size=2,
        in_chans=3,
        embed_dim=768,
        decoder_embed_dim=1024,
        depth=12,  
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        out_layers=None,
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
        
        logger.info(f"Initializing VisionTransformerDecoder with config:")
        logger.info(f"Image size: {img_size}, Patch size: {patch_size}")
        logger.info(f"Num frames: {num_frames}, Tubelet size: {tubelet_size}")
        logger.info(f"Input channels: {in_chans}, Embed dim: {embed_dim}")
        logger.info(f"Decoder embed dim: {decoder_embed_dim}, Depth: {depth}")
        logger.info(f"Num heads: {num_heads}, MLP ratio: {mlp_ratio}")
        
        # Dimensions
        self.img_size = img_size
        self.patch_size = patch_size
        self.input_size = img_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size
        self.in_chans = in_chans
        self.uniform_power = uniform_power
        
        # Calculate number of patches
        self.is_video = num_frames > 1
        grid_size = self.input_size // self.patch_size
        grid_depth = self.num_frames // self.tubelet_size

        if self.is_video:
            self.num_patches = num_patches = (
                (num_frames // tubelet_size) 
                * (img_size // patch_size) 
                * (img_size // patch_size)
            )
            logger.info(f"Video mode: num_patches = {num_patches} ({num_frames // tubelet_size} x {img_size // patch_size} x {img_size // patch_size})")
        else:
            self.num_patches = num_patches = (
                (img_size // patch_size)
                * (img_size // patch_size)
            )
            logger.info(f"Image mode: num_patches = {num_patches} ({img_size // patch_size} x {img_size // patch_size})")
            
        # Initial projection from encoder embedding dim to decoder embedding dim
        self.embed_proj = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        # self.embed_proj = nn.Sequential(
        #     nn.Linear(embed_dim, decoder_embed_dim),
        #     nn.GELU(),
        #     nn.Linear(decoder_embed_dim, decoder_embed_dim)
        # )

        # Attention Blocks
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                act_layer=nn.GELU,
                grid_size=grid_size,
                grid_depth=grid_depth,
                attn_drop=attn_drop_rate,
                norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        
        # Nonlinear projection to patch space
        patch_dim = patch_size * patch_size * tubelet_size * in_chans

        # Replace linear patch_proj with ConvTranspose layers
        if self.is_video:
            self.patch_proj = nn.Sequential(
                # First project to intermediate dimension
                nn.Linear(decoder_embed_dim * 2, patch_dim),
                # Reshape layer will be in forward pass
                nn.ConvTranspose3d(
                    in_channels=patch_dim // (patch_size * patch_size * tubelet_size),  # Should equal in_chans
                    out_channels=in_chans,
                    kernel_size=(tubelet_size, patch_size, patch_size),
                    stride=(tubelet_size, patch_size, patch_size)
                )
            )
        else:
            self.patch_proj = nn.Sequential(
                # First project to intermediate dimension
                nn.Linear(decoder_embed_dim * 2, patch_dim),
                # Reshape layer will be in forward pass
                nn.ConvTranspose2d(
                    in_channels=patch_dim // (patch_size * patch_size),  # Should equal in_chans
                    out_channels=in_chans,
                    kernel_size=(patch_size, patch_size),
                    stride=(patch_size, patch_size)
                )
            )
        # self.patch_proj = nn.Sequential(
        #     nn.Linear(decoder_embed_dim, decoder_embed_dim * 2),
        #     nn.GELU(),
        #     nn.Linear(decoder_embed_dim * 2, decoder_embed_dim),
        #     nn.GELU(),
        #     nn.Linear(decoder_embed_dim, patch_dim),
        # )

        # Positional embeddings
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, decoder_embed_dim),
            requires_grad=False
        )
        if self.pos_embed is not None:
            self._init_pos_embed(self.pos_embed.data)
        self.init_std = init_std
        self.apply(self._init_weights)
        self._rescale_blocks()
        
    def _init_pos_embed(self, pos_embed):
        """
        Initialize the positional embedding tensor with sine-cosine positional encodings.

        Args:
            pos_embed (torch.Tensor): The positional embedding tensor to be initialized.
                The last dimension of `pos_embed` should match the embedding dimension.

        The function computes either 2D or 3D sine-cosine positional encodings based on
        whether the model is processing video input. For video inputs, the encoding is 3D
        with dimensions determined by the grid size and depth. For image inputs, the 
        encoding is 2D based only on the grid size.

        The computed positional encodings are copied to the `pos_embed` tensor.
        """
        embed_dim = pos_embed.size(-1)
        grid_size = self.input_size // self.patch_size
        
        logger.debug(f"Initializing positional embeddings:")
        logger.debug(f"Embed dim: {embed_dim}, Grid size: {grid_size}")
        
        if self.is_video:
            grid_depth = self.num_frames // self.tubelet_size
            logger.debug(f"Video mode: Grid depth = {grid_depth}")
            sincos = get_3d_sincos_pos_embed(
                embed_dim,
                grid_size,
                grid_depth,
                cls_token=False,
                uniform_power=self.uniform_power
            )
        else:
            logger.debug("Image mode: Using 2D positional embeddings")
            sincos = get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False)
            
        logger.debug(f"Positional embedding shape: {sincos.shape}")
        pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def _init_weights(self, m):
        """
        Initialize the weights of the given module.

        Args:
            m (nn.Module): The module whose weights and biases are to be initialized.

        This function initializes the weights of linear, layer normalization, and
        convolutional layers (both 2D and 3D) using a truncated normal distribution
        with a standard deviation specified by `self.init_std`. It also sets the
        biases of these layers to zero if present. For nn.LayerNorm, the weights
        are initialized to 1.0 and biases to 0.
        """

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv3d):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _rescale_blocks(self):
        """
        Rescale the weights of each layer in the VisionTransformer by a factor of
        sqrt(2 * layer_id), following the recipe in the VisionTransformer paper.
        This is used to prevent the weights from growing too large during training.
        """
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def interpolate_pos_encoding(self, x, pos_embed):

        """
        Interpolate the positional encoding to match the input dimensions.

        Args:
            x (torch.Tensor): The input tensor, which is either a video or an image.
                For video input, it should have shape (B, C, T, H, W).
                For image input, it should have shape (B, C, H, W).
            pos_embed (torch.Tensor): The positional embedding tensor to be interpolated.
                It has shape (1, N, dim), where N is the number of patches and dim is the
                embedding dimension.

        Returns:
            torch.Tensor: The interpolated positional embedding tensor with adjusted dimensions
            to match the input size.

        The function handles both video and image inputs, performing trilinear interpolation
        for video positional encodings and bicubic interpolation for image positional encodings.
        It first checks if the current positional embedding already matches the input dimensions.
        If not, it computes the scale factor based on the input size and the initialized shape
        of the positional embedding, and interpolates accordingly.
        """

        _, N, dim = pos_embed.shape

        if self.is_video:

            # If pos_embed already corret size, just return
            _, _, T, H, W = x.shape
            if H == self.input_size and W == self.input_size and T == self.num_frames:
                return pos_embed

            # Convert depth, height, width of input to be measured in patches
            # instead of pixels/frames
            T = T // self.tubelet_size
            H = H // self.patch_size
            W = W // self.patch_size

            # Compute the initialized shape of the positional embedding measured
            # in patches
            N_t = self.num_frames // self.tubelet_size
            N_h = N_w = self.input_size // self.patch_size
            assert N_h * N_w * N_t == N, 'Positional embedding initialized incorrectly'

            # Compute scale factor for spatio-temporal interpolation
            scale_factor = (T/N_t, H/N_h, W/N_w)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, N_t, N_h, N_w, dim).permute(0, 4, 1, 2, 3),
                scale_factor=scale_factor,
                mode='trilinear')
            pos_embed = pos_embed.permute(0, 2, 3, 4, 1).view(1, -1, dim)
            return pos_embed

        else:

            # If pos_embed already corret size, just return
            _, _, H, W = x.shape
            if H == self.input_size and W == self.input_size:
                return pos_embed

            # Compute scale factor for spatial interpolation
            npatch = (H // self.patch_size) * (W // self.patch_size)
            scale_factor = math.sqrt(npatch / N)

            pos_embed = nn.functional.interpolate(
                pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
                scale_factor=scale_factor,
                mode='bicubic')
            pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
            return pos_embed
        

    def forward(self, ctxts, tgts, masks_ctxts, masks_tgts, return_intermediates=False):
        """
        Forward pass through decoder
        Args:
            x: Single tensor or list of tensors of shape (B, N, embed_dim)
            return_intermediates: Whether to return intermediate activations
        """
        if isinstance(ctxts, list):
            logger.debug(f"Processing list of {len(ctxts)} embeddings")
            outputs = []
            for i, (ctxt, tgt, masks_ctxt, masks_tgt) in enumerate(zip(ctxts, tgts, masks_ctxts, masks_tgts)):
                logger.debug(f"Processing embedding {i+1}/{len(ctxts)}")
                if return_intermediates:
                    out, intermediates = self._forward_single(ctxt, tgt, masks_ctxt, masks_tgt, return_intermediates=True)
                    outputs.append((out, intermediates))
                else:
                    outputs.append(self._forward_single(ctxt, tgt, masks_ctxt, masks_tgt))
            return outputs
        else:
            logger.debug("Processing single embedding")
            return self._forward_single(ctxts, tgts, masks_ctxts, masks_tgts, return_intermediates)

    def _forward_single(self, ctxt, tgt, masks_ctxt, masks_tgt, return_intermediates=False):
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
        B = len(ctxt) // len(masks_ctxt)
        logger.debug(f"Forward pass - Batch size: {B}")
        logger.debug(f"Context shape: {ctxt.shape}, Target shape: {tgt.shape}")
        logger.debug(f"Context masks shape: {masks_ctxt[0].shape}, Target masks shape: {masks_tgt[0].shape}")
        
        # Project to decoder embedding dimension
        ctxt_tokens = self.embed_proj(ctxt)
        tgt_tokens = self.embed_proj(tgt)
        logger.debug(f"After projection - Context tokens: {ctxt_tokens.shape}, Target tokens: {tgt_tokens.shape}")

        # Add positional embedding
        if self.pos_embed is not None:
            ctxt_pos_embed = self.interpolate_pos_encoding(ctxt_tokens, self.pos_embed)
            ctxt_tokens += apply_masks(ctxt_pos_embed, masks_ctxt)
            
            tgt_pos_embed = self.interpolate_pos_encoding(tgt_tokens, self.pos_embed)
            tgt_tokens += apply_masks(tgt_pos_embed, masks_tgt)
            logger.debug("Added positional embeddings to tokens")

        # Concatenate tokens
        x = torch.cat([ctxt_tokens, tgt_tokens], dim=1)
        logger.debug(f"Concatenated tokens shape: {x.shape}")

        # Process masks
        masks_ctxt = torch.cat(masks_ctxt, dim=0)
        masks_tgt = torch.cat(masks_tgt, dim=0)
        masks = torch.cat([masks_ctxt, masks_tgt], dim=1)
        logger.debug(f"Combined masks shape: {masks.shape}")

        # Process through transformer blocks
        intermediates = []
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=masks)
            if return_intermediates:
                intermediates.append(x)
            logger.debug(f"Block {i+1}/{len(self.blocks)} output shape: {x.shape}")

        if self.norm is not None:
            x = self.norm(x)
            logger.debug(f"After normalization shape: {x.shape}")

        # Project to patch dimension
        x = self.patch_proj[0](x)
        logger.debug(f"After initial projection shape: {x.shape}")

        # Reshape and apply transposed convolution
        if self.num_frames > 1:
            B, N, C = x.shape
            T = self.num_frames // self.tubelet_size
            H = W = self.img_size // self.patch_size
            
            x = x.reshape(B, T, H, W, -1)
            x = x.permute(0, 4, 1, 2, 3)
            logger.debug(f"Reshaped for 3D conv shape: {x.shape}")
            
            x = self.patch_proj[1](x)
            logger.debug(f"Final output shape (video): {x.shape}")
        else:
            B, N, C = x.shape
            H = W = self.img_size // self.patch_size
            
            x = x.reshape(B, H, W, -1)
            x = x.permute(0, 3, 1, 2)
            logger.debug(f"Reshaped for 2D conv shape: {x.shape}")
            
            x = self.patch_proj[1](x)
            logger.debug(f"Final output shape (image): {x.shape}")

        if return_intermediates:
            return x, intermediates
        return x

def vit_tiny(patch_size=16, **kwargs):
    return VisionTransformerDecoder(
        patch_size=patch_size, 
        embed_dim=192, 
        depth=12, 
        num_heads=3, 
        mlp_ratio=4,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs
    )      

def vit_small(patch_size=16, **kwargs):
    return VisionTransformerDecoder(
        patch_size=patch_size, 
        embed_dim=384, 
        depth=12, 
        num_heads=6, 
        mlp_ratio=4,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs
    )

def vit_base(patch_size=16, **kwargs):
    return VisionTransformerDecoder(
        patch_size=patch_size, 
        embed_dim=768, 
        depth=12, 
        num_heads=12, 
        mlp_ratio=4,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs
    )

def vit_large(patch_size=16, **kwargs):
    return VisionTransformerDecoder(
        patch_size=patch_size, 
        embed_dim=1024, 
        depth=24, 
        num_heads=16, 
        mlp_ratio=4,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs
    )

def vit_huge(patch_size=16, **kwargs):
    return VisionTransformerDecoder(
        patch_size=patch_size, 
        embed_dim=1280, 
        depth=32, 
        num_heads=16, 
        mlp_ratio=4,
        qkv_bias=True, 
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        **kwargs
    )