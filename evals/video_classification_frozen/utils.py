# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms

from src.datasets.utils.video.randerase import RandomErasing

from src.models.utils.pos_embs import get_1d_sincos_pos_embed
from src.masks.utils import apply_masks


class FrameAggregation(nn.Module):
    """
    Process each frame independently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        max_frames=10000,
        use_pos_embed=False,
        attend_across_segments=False
    ):
        """
        Args:
            model: The model to be wrapped.
            max_frames: The maximum number of frames in a video.
            use_pos_embed: Whether to use 1D-temporal position embeddings.
            attend_across_segments: Whether to attend across segments using cross-attention.
        """
        super().__init__()
        self.model = model
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.attend_across_segments = attend_across_segments
        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_frames, embed_dim),
                requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_frames)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):

        # TODO: implement attend_across_segments=False
        # num_clips = len(x)
        """
        Process each frame independently and concatenate all tokens.

        Args:
            x (list of Tensors): list of num_clips tensors of shape [B, C, T, H, W]
            clip_indices (list of Tensors or None): indices of patch tokens to mask (remove)

        Returns:
            list of Tensors: list of num_views_per_clip tensors of shape [B, T*N, D]
        """
        num_views_per_clip = len(x[0])

        # Concatenate views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        # Concatenate clips along temporal dimension
        x = torch.cat(x, dim=2)
        B, C, T, H, W = x.size()

        # Put each frame along the batch dimension
        x = x.permute(0, 2, 1, 3, 4).reshape(B*T, C, H, W)

        outputs = self.model(x)
        _, N, D = outputs.size()
        outputs = outputs.reshape(B, T, N, D).flatten(1, 2)

        # Separate views into list
        B = B // num_views_per_clip
        all_outputs = []
        for i in range(num_views_per_clip):
            o = outputs[i*B:(i+1)*B]
            # Compute positional embedding
            if (self.pos_embed is not None) and (clip_indices is not None):
                pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, F, D]
                pos_embed = apply_masks(pos_embed, clip_indices, concat=False)  # list(Tensor([B, T, D]))
                pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, N, 1)  # [B, T*num_clips, N, D]
                pos_embed = pos_embed.flatten(1, 2)
                o += pos_embed
            all_outputs += [o]

        return all_outputs


class ClipAggregation(nn.Module):
    """
    Process each clip independently and concatenate all tokens
    """

    def __init__(
        self,
        model,
        tubelet_size=2,
        max_frames=10000,
        use_pos_embed=False,
        attend_across_segments=False
    ):
        """
        Initialize ClipAggregation module.

        Args:
            model (nn.Module): base Vision Transformer model
            tubelet_size (int, optional): the size of each tubelet. Defaults to 2.
            max_frames (int, optional): maximum number of frames to process. Defaults to 10000.
            use_pos_embed (bool, optional): whether to use positional embedding. Defaults to False.
            attend_across_segments (bool, optional): whether to attend across segments. Defaults to False.
        """
        super().__init__()
        self.model = model
        self.tubelet_size = tubelet_size
        self.embed_dim = embed_dim = model.embed_dim
        self.num_heads = model.num_heads
        self.attend_across_segments = attend_across_segments
        # 1D-temporal pos-embedding
        self.pos_embed = None
        if use_pos_embed:
            max_T = max_frames // tubelet_size
            self.pos_embed = nn.Parameter(
                torch.zeros(1, max_T, embed_dim),
                requires_grad=False)
            sincos = get_1d_sincos_pos_embed(embed_dim, max_T)
            self.pos_embed.copy_(torch.from_numpy(sincos).float().unsqueeze(0))

    def forward(self, x, clip_indices=None):

        """
        Forward pass for processing video clips through the Vision Transformer model.

        Args:
            x (list of list of torch.Tensor): A nested list where each inner list contains
                tensors representing different spatial and temporal views of a video clip.
                Each tensor has shape (B, C, T, H, W).
            clip_indices (list of torch.Tensor, optional): Indices of clips to be used 
                for positional embedding. Each tensor should have shape (B, T // tubelet_size).

        Returns:
            list of torch.Tensor: If `attend_across_segments` is False, returns a list of
                outputs for each view. If True, returns concatenated temporal outputs with
                positional embeddings applied, for each view, with each tensor shaped as 
                (B, T*num_clips*N, D).
        """

        num_clips = len(x)
        num_views_per_clip = len(x[0])
        B, C, T, H, W = x[0][0].size()

        # Concatenate all spatial and temporal views along batch dimension
        x = [torch.cat(xi, dim=0) for xi in x]
        x = torch.cat(x, dim=0)
        outputs = self.model(x)
        _, N, D = outputs.size()

        T = T // self.tubelet_size  # Num temporal tokens
        N = N // T  # Num spatial tokens

        # Unroll outputs into a 2D array [spatial_views x temporal_views]
        eff_B = B * num_views_per_clip
        all_outputs = [[] for _ in range(num_views_per_clip)]
        for i in range(num_clips):
            o = outputs[i*eff_B:(i+1)*eff_B]
            for j in range(num_views_per_clip):
                all_outputs[j].append(o[j*B:(j+1)*B])

        if not self.attend_across_segments:
            return all_outputs

        for i, outputs in enumerate(all_outputs):

            # Concatenate along temporal dimension
            outputs = [o.reshape(B, T, N, D) for o in outputs]
            outputs = torch.cat(outputs, dim=1).flatten(1, 2)

            # Compute positional embedding
            if (self.pos_embed is not None) and (clip_indices is not None):
                clip_indices = [c[:, ::self.tubelet_size] for c in clip_indices]
                pos_embed = self.pos_embed.repeat(B, 1, 1)  # [B, F, D]
                pos_embed = apply_masks(pos_embed, clip_indices, concat=False)  # list(Tensor([B, T, D]))
                pos_embed = torch.cat(pos_embed, dim=1)  # concatenate along temporal dimension
                pos_embed = pos_embed.unsqueeze(2).repeat(1, 1, N, 1)  # [B, T*num_clips, N, D]
                pos_embed = pos_embed.flatten(1, 2)
                outputs += pos_embed

            all_outputs[i] = outputs

        return all_outputs


def make_transforms(
    training=True,
    random_horizontal_flip=True,
    random_resize_aspect_ratio=(3/4, 4/3),
    random_resize_scale=(0.3, 1.0),
    reprob=0.0,
    auto_augment=False,
    motion_shift=False,
    crop_size=224,
    num_views_per_clip=1,
    normalize=((0.485, 0.456, 0.406),
               (0.229, 0.224, 0.225))
):

    """
    Create VideoTransform or EvalVideoTransform depending on the arguments.

    Args:
        training (bool): Whether to create a VideoTransform or EvalVideoTransform.
        random_horizontal_flip (bool): Whether to randomly flip horizontally.
        random_resize_aspect_ratio (tuple): Range of aspect ratios to randomly resize.
        random_resize_scale (tuple): Range of scales to randomly resize.
        reprob (float): Probability of random erasing.
        auto_augment (bool): Whether to use RandAugment.
        motion_shift (bool): Whether to perform motion shift.
        crop_size (int): Short side size of the crop.
        num_views_per_clip (int): Number of views per clip.
        normalize (tuple): Mean and std of the normalization.

    Returns:
        VideoTransform or EvalVideoTransform.
    """
    if not training and num_views_per_clip > 1:
        print('Making EvalVideoTransform, multi-view')
        _frames_augmentation = EvalVideoTransform(
            num_views_per_clip=num_views_per_clip,
            short_side_size=crop_size,
            normalize=normalize,
        )

    else:
        _frames_augmentation = VideoTransform(
            training=training,
            random_horizontal_flip=random_horizontal_flip,
            random_resize_aspect_ratio=random_resize_aspect_ratio,
            random_resize_scale=random_resize_scale,
            reprob=reprob,
            auto_augment=auto_augment,
            motion_shift=motion_shift,
            crop_size=crop_size,
            normalize=normalize,
        )
    return _frames_augmentation


class VideoTransform(object):

    def __init__(
        self,
        training=True,
        random_horizontal_flip=True,
        random_resize_aspect_ratio=(3/4, 4/3),
        random_resize_scale=(0.3, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=224,
        normalize=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
    ):

        """
        Initialize VideoTransform class with various transformation parameters.

        Args:
            training (bool): Whether to perform training transforms.
            random_horizontal_flip (bool): Whether to randomly flip horizontally.
            random_resize_aspect_ratio (tuple): Range of aspect ratios to randomly resize.
            random_resize_scale (tuple): Range of scales to randomly resize.
            reprob (float): Probability of random erasing.
            auto_augment (bool): Whether to use RandAugment.
            motion_shift (bool): Whether to perform motion shift.
            crop_size (int): Short side size of the crop.
            normalize (tuple): Mean and standard deviation for normalization.
        """
        self.training = training

        short_side_size = int(crop_size * 256 / 224)
        self.eval_transform = video_transforms.Compose([
            video_transforms.Resize(short_side_size, interpolation='bilinear'),
            video_transforms.CenterCrop(size=(crop_size, crop_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])

        self.random_horizontal_flip = random_horizontal_flip
        self.random_resize_aspect_ratio = random_resize_aspect_ratio
        self.random_resize_scale = random_resize_scale
        self.auto_augment = auto_augment
        self.motion_shift = motion_shift
        self.crop_size = crop_size
        self.normalize = torch.tensor(normalize)

        self.autoaug_transform = video_transforms.create_random_augment(
            input_size=(crop_size, crop_size),
            auto_augment='rand-m7-n4-mstd0.5-inc1',
            interpolation='bicubic',
        )

        self.spatial_transform = video_transforms.random_resized_crop_with_shift \
            if motion_shift else video_transforms.random_resized_crop

        self.reprob = reprob
        self.erase_transform = RandomErasing(
            reprob,
            mode='pixel',
            max_count=1,
            num_splits=1,
            device='cpu',
        )

    def __call__(self, buffer):

        """
        Apply training or evaluation transformations to a video buffer.

        If the instance is in evaluation mode, apply the evaluation transform
        and return the transformed buffer. Otherwise, perform a series of 
        transformations including converting frames to PIL images, applying 
        auto-augmentation if enabled, converting images to tensors, normalizing,
        spatially transforming the buffer, and optionally applying horizontal 
        flip and random erasing.

        Args:
            buffer (list): A list of video frames.

        Returns:
            list: A list containing the transformed video buffer.
        """

        if not self.training:
            return [self.eval_transform(buffer)]

        buffer = [transforms.ToPILImage()(frame) for frame in buffer]

        if self.auto_augment:
            buffer = self.autoaug_transform(buffer)

        buffer = [transforms.ToTensor()(img) for img in buffer]
        buffer = torch.stack(buffer)  # T C H W
        buffer = buffer.permute(0, 2, 3, 1)  # T H W C

        buffer = tensor_normalize(buffer, self.normalize[0], self.normalize[1])
        buffer = buffer.permute(3, 0, 1, 2)  # T H W C -> C T H W

        buffer = self.spatial_transform(
            images=buffer,
            target_height=self.crop_size,
            target_width=self.crop_size,
            scale=self.random_resize_scale,
            ratio=self.random_resize_aspect_ratio,
        )
        if self.random_horizontal_flip:
            buffer, _ = video_transforms.horizontal_flip(0.5, buffer)

        if self.reprob > 0:
            buffer = buffer.permute(1, 0, 2, 3)
            buffer = self.erase_transform(buffer)
            buffer = buffer.permute(1, 0, 2, 3)

        return [buffer]


class EvalVideoTransform(object):

    def __init__(
        self,
        num_views_per_clip=1,
        short_side_size=224,
        normalize=((0.485, 0.456, 0.406),
                   (0.229, 0.224, 0.225))
    ):
        """
        Initialize EvalVideoTransform with parameters for processing video clips.

        Args:
            num_views_per_clip (int): Number of spatial views to sample per video clip.
            short_side_size (int): Target size for the short side after resizing.
            normalize (tuple): Mean and standard deviation for normalization, with
                            mean as the first element and std as the second element.
        """

        self.views_per_clip = num_views_per_clip
        self.short_side_size = short_side_size
        self.spatial_resize = video_transforms.Resize(short_side_size, interpolation='bilinear')
        self.to_tensor = video_transforms.Compose([
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=normalize[0], std=normalize[1])
        ])

    def __call__(self, buffer):

        # Sample several spatial views of each clip
        """
        Resizes the input video clip to a fixed short side size and then sample
        multiple spatial views from it.

        Args:
            buffer (ndarray or tensor): Input video clip of shape (T, H, W, C)

        Returns:
            list of tensors: A list of length `self.views_per_clip`, each element
                is a tensor of shape (C, T, H', W') where H' and W' are the height
                and width of the resized video.
        """
        buffer = np.array(self.spatial_resize(buffer))
        T, H, W, C = buffer.shape

        num_views = self.views_per_clip
        side_len = self.short_side_size
        spatial_step = (max(H, W) - side_len) // (num_views - 1)

        all_views = []
        for i in range(num_views):
            start = i*spatial_step
            if H > W:
                view = buffer[:, start:start+side_len, :, :]
            else:
                view = buffer[:, :, start:start+side_len, :]
            view = self.to_tensor(view)
            all_views.append(view)

        return all_views


def tensor_normalize(tensor, mean, std):
    """
    Normalize a given tensor by subtracting the mean and dividing the std.
    Args:
        tensor (tensor): tensor to normalize.
        mean (tensor or list): mean value to subtract.
        std (tensor or list): std to divide.
    """
    if tensor.dtype == torch.uint8:
        tensor = tensor.float()
        tensor = tensor / 255.0
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor - mean
    tensor = tensor / std
    return tensor
