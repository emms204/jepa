# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch.nn as nn


class MultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        """
        Initialize a MultiMaskWrapper module.

        Parameters
        ----------
        backbone : Module
            The module to be wrapped. It should have a forward method that takes
            two arguments: x and masks. masks is an optional argument that defaults
            to None.
        """
        super().__init__()
        self.backbone = backbone

    def forward(self, x, masks=None):
        """
        Forward pass through the module.

        Parameters
        ----------
        x : Tensor
            The input tensor.
        masks : None or Tensor or list of Tensor
            The masks to apply to the input. If None, no masking is applied.
            If a single tensor, it is applied to all input tensors.
            If a list of tensors, each mask is applied to the corresponding
            input tensor.

        Returns
        -------
        out : Tensor or list of Tensor
            The output tensors. If masks is None or a single tensor, a single
            tensor is returned. Otherwise, a list of tensors is returned, each
            corresponding to the output of one of the masks.
        """
        if masks is None:
            return self.backbone(x)

        if (masks is not None) and not isinstance(masks, list):
            masks = [masks]
        outs = []
        for m in masks:
            outs += [self.backbone(x, masks=m)]
        return outs


class PredictorMultiMaskWrapper(nn.Module):

    def __init__(self, backbone):
        """
        Initialize a PredictorMultiMaskWrapper module.

        Parameters
        ----------
        backbone : Module
            The module to be wrapped. It should have a forward method that takes
            four arguments: ctxt, tgt, masks_ctxt, and masks_tgt.
        """

        super().__init__()
        self.backbone = backbone

    def forward(self, ctxt, tgt, masks_ctxt, masks_tgt):
        """
        Forward pass through the module.

        Parameters
        ----------
        ctxt : Tensor or list of Tensor
            The context tensors.
        tgt : Tensor or list of Tensor
            The target tensors.
        masks_ctxt : None or Tensor or list of Tensor
            The masks for the context tensors. If None, no masking is applied.
            If a single tensor, it is applied to all context tensors.
            If a list of tensors, each mask is applied to the corresponding
            context tensor.
        masks_tgt : None or Tensor or list of Tensor
            The masks for the target tensors. If None, no masking is applied.
            If a single tensor, it is applied to all target tensors.
            If a list of tensors, each mask is applied to the corresponding
            target tensor.

        Returns
        -------
        out : list of Tensor
            The output tensors, each corresponding to the output of one of the
            masks.
        """
        if type(ctxt) is not list:
            ctxt = [ctxt]
        if type(tgt) is not list:
            tgt = [tgt]
        if type(masks_ctxt) is not list:
            masks_ctxt = [masks_ctxt]
        if type(masks_tgt) is not list:
            masks_tgt = [masks_tgt]

        outs = []
        for i, (zi, hi, mc, mt) in enumerate(zip(ctxt, tgt, masks_ctxt, masks_tgt)):
            outs += [self.backbone(zi, hi, mc, mt, mask_index=i)]
        return outs
