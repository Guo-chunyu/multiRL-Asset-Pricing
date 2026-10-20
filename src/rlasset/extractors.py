"""Shared temporal encoder inspired by RLPortfolio's convolutional EIIE.

This is a feature extractor for SB3 actors/critics, not a reproduction of EIIE's
complete policy or policy-gradient learning procedure.
"""
from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class SharedTemporalExtractor(BaseFeaturesExtractor):
    """Apply identical temporal filters per asset; append drifted holdings.

    Input: ``prices[B, F, N, L]`` and ``weights[B, N + 1]`` (cash first).
    Output: asset encodings in ticker order followed by the holdings vector.
    The shared encoder is asset-permutation equivariant; the subsequent SB3
    dense actor/critic need not have that property.
    """

    def __init__(self, observation_space: spaces.Dict, channels: int = 16):
        if not isinstance(observation_space, spaces.Dict):
            raise ValueError("SharedTemporalExtractor requires a Dict observation")
        if "prices" not in observation_space.spaces or "weights" not in observation_space.spaces:
            raise ValueError("observation must contain prices and weights")
        price_shape = observation_space["prices"].shape
        if len(price_shape) != 3:
            raise ValueError("prices must have shape (features, assets, window)")
        n_features, n_assets, window = price_shape
        if min(n_features, n_assets, window, channels) < 1:
            raise ValueError("prices dimensions and channels must be positive")
        if observation_space["weights"].shape != (n_assets + 1,):
            raise ValueError("weights must contain cash plus one weight per asset")
        super().__init__(observation_space, features_dim=n_assets * channels + n_assets + 1)
        self.n_assets = n_assets
        self.channels = channels
        kernel_size = min(3, window)
        self.encoder = nn.Sequential(
            nn.Conv1d(n_features, channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.Conv1d(channels, channels, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        prices = observations["prices"].float()
        batch, features, assets, window = prices.shape
        asset_series = prices.permute(0, 2, 1, 3).reshape(batch * assets, features, window)
        encoded = self.encoder(asset_series).reshape(batch, assets * self.channels)
        return torch.cat([encoded, observations["weights"].float()], dim=1)
