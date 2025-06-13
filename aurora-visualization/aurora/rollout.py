"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import dataclasses
from typing import Generator

import torch

from aurora.batch import Batch
from aurora.model.aurora import Aurora

__all__ = ["rollout"]

def rollout(model: Aurora, batch: Batch, steps: int) -> Generator[Batch, None, None]:
    """Perform a roll-out to make long-term predictions."""

    # Ensure batch is preprocessed
    batch = model.batch_transform_hook(batch)
    p = next(model.parameters())
    batch = batch.type(p.dtype).crop(model.patch_size).to(p.device)

    for _ in range(steps):
        pred = model.forward(batch)
        yield pred

        # Correct slicing on channel dimension, not timestep
        surf_vars_sliced = {k: v[:, :, :1] for k, v in pred.surf_vars.items()}
        atmos_vars_sliced = {k: v[:, :, :1] for k, v in pred.atmos_vars.items()}

        print("Before slice:", {k: v.shape for k, v in pred.atmos_vars.items()})
        print("After slice:", {k: v.shape for k, v in atmos_vars_sliced.items()})

        batch = dataclasses.replace(
            pred,
            surf_vars={
                k: torch.cat([batch.surf_vars[k][:, :, 1:], surf_vars_sliced[k]], dim=2)
                for k in surf_vars_sliced
            },
            atmos_vars={
                k: torch.cat([batch.atmos_vars[k][:, :, 1:], atmos_vars_sliced[k]], dim=2)
                for k in atmos_vars_sliced
            },
        )