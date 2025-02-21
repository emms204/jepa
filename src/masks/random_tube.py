# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from multiprocessing import Value

from logging import getLogger

import torch
import numpy as np

_GLOBAL_SEED = 0
logger = getLogger()


class MaskCollator(object):

    def __init__(
        self,
        cfgs_mask,
        crop_size=(224, 224),
        num_frames=16,
        patch_size=(16, 16),
        tubelet_size=2,
    ):
        """
        Initialize the MaskCollator with a list of config dictionaries.

        Args:
            cfgs_mask (list): A list of config dictionaries for the mask
                generators. Each config should contain a 'ratio' key with the
                ratio of patches to remove.
            crop_size (tuple): The spatial size of the cropped input data.
            num_frames (int): The number of frames in the input data.
            patch_size (tuple): The size of each patch in the input data.
            tubelet_size (int): The size of each tubelet in the input data.
        """
        super(MaskCollator, self).__init__()

        self.mask_generators = []
        for m in cfgs_mask:
            mask_generator = _MaskGenerator(
                crop_size=crop_size,
                num_frames=num_frames,
                spatial_patch_size=patch_size,
                temporal_patch_size=tubelet_size,
                ratio=m.get('ratio'),
            )
            self.mask_generators.append(mask_generator)

    def step(self):
        for mask_generator in self.mask_generators:
            mask_generator.step()

    def __call__(self, batch):

        """
        Collates a batch of samples and generates a set of masks for each sample in the batch.

        Args:
            batch (list): A list of samples to be collated.

        Returns:
            tuple: A tuple containing:
                - collated_batch (torch.Tensor): The collated batch of samples.
                - collated_masks_enc (list of torch.Tensor): A list of tensors, each containing indices
                of context tokens for each mask generator.
                - collated_masks_pred (list of torch.Tensor): A list of tensors, each containing indices
                of target tokens for each mask generator.
        """

        batch_size = len(batch)
        collated_batch = torch.utils.data.default_collate(batch)

        collated_masks_pred, collated_masks_enc = [], []
        for i, mask_generator in enumerate(self.mask_generators):
            masks_enc, masks_pred = mask_generator(batch_size)
            collated_masks_enc.append(masks_enc)
            collated_masks_pred.append(masks_pred)

        return collated_batch, collated_masks_enc, collated_masks_pred


class _MaskGenerator(object):

    def __init__(
        self,
        crop_size=(224, 224),
        num_frames=16,
        spatial_patch_size=(16, 16),
        temporal_patch_size=2,
        ratio=0.9,
    ):
        """
        Initialize a _MaskGenerator object.

        Args:
            crop_size (tuple or int): The size of the cropped input data.
            num_frames (int): The number of frames in the input data.
            spatial_patch_size (tuple): The size of each patch in the input data.
            temporal_patch_size (int): The size of each tubelet in the input data.
            ratio (float): The ratio of patches to remove.
        """
        super(_MaskGenerator, self).__init__()
        if not isinstance(crop_size, tuple):
            crop_size = (crop_size, ) * 2
        self.crop_size = crop_size
        self.height, self.width = crop_size[0] // spatial_patch_size, crop_size[1] // spatial_patch_size
        self.duration = num_frames // temporal_patch_size

        self.spatial_patch_size = spatial_patch_size
        self.temporal_patch_size = temporal_patch_size
        self.num_patches_spatial = self.height*self.width

        self.ratio = ratio

        self.num_keep_spatial = int(self.num_patches_spatial*(1.-self.ratio))
        self.num_keep = self.num_keep_spatial * self.duration

        self._itr_counter = Value('i', -1)  # collator is shared across worker processes

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch_size):
        """
        Generate a list of masks for each sample in the batch. The masks are used
        to remove patches from the input data. The number of patches to remove
        is determined by the `ratio` parameter. The patches are randomly selected
        from the spatial and temporal dimensions of the input data.

        Args:
            batch_size (int): The batch size.

        Returns:
            collated_masks_enc (list of torch.Tensor): A list of tensors, each containing indices
                of context tokens for each mask generator.
            collated_masks_pred (list of torch.Tensor): A list of tensors, each containing indices
                of target tokens for each mask generator.
        """
        def sample_mask():
            mask = np.hstack([
                np.zeros(self.num_patches_spatial - self.num_keep_spatial),
                np.ones(self.num_keep_spatial),
            ])
            np.random.shuffle(mask)
            mask = torch.tensor(np.tile(mask, (self.duration, 1)))
            mask = mask.flatten()
            mask_p = torch.argwhere(mask == 0).squeeze()
            mask_e = torch.nonzero(mask).squeeze()
            return mask_e, mask_p

        collated_masks_pred, collated_masks_enc = [], []
        for _ in range(batch_size):
            mask_e, mask_p = sample_mask()
            collated_masks_enc.append(mask_e)
            collated_masks_pred.append(mask_p)

        collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
        collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)

        return collated_masks_enc, collated_masks_pred
