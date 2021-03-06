#!/usr/bin/env python

import torch as th
import torch.nn as nn
from pathlib import Path
from .feature import FeatureExtractor


class Executor(nn.Module):
    """
    Executor is a class to handle feature extraction 
    and forward process of the separation networks.
    """
    def __init__(self, nnet, extractor_kwargs=None, get_mask=True):
        super(Executor, self).__init__()
        self.nnet = nnet
        self.extractor = FeatureExtractor(
            **extractor_kwargs) if extractor_kwargs else None
        self.frame_len = extractor_kwargs['frame_len'] if extractor_kwargs else None
        self.frame_hop = extractor_kwargs['frame_hop'] if extractor_kwargs else None
        self.get_mask = get_mask

    def resume(self, checkpoint):
        """
        Resume from checkpoint
        """
        if not Path(checkpoint).exists():
            raise FileNotFoundError(
                f"Could not find resume checkpoint: {checkpoint}")
        cpt = th.load(checkpoint, map_location="cpu")
        self.load_state_dict(cpt["model_state_dict"])
        return cpt["epoch"]

    def _compute_feats(self, egs):
        """
        Compute features: N x F x T
        """
        if not self.extractor:
            raise RuntimeError("self.extractor is None, "
                               "do not need to compute features")
        mag, pha, f = self.extractor(egs["mix"])
        return mag, pha, f

    def forward(self, egs):
        mag, pha, f = self._compute_feats(egs)
        out = self.nnet(f)
        if self.get_mask:
            return out
        else:
            return [self.extractor.istft(m * mag, pha) for m in out]
