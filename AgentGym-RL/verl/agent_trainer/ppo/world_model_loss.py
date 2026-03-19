"""Helpers for world-model auxiliary loss on observation tokens."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

import verl.utils.torch_functional as verl_F


def compute_observation_mask(
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    observation_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Resolve observation-token mask for the response region.

    If an explicit ``observation_mask`` is already carried by the batch we use
    it directly; otherwise we fall back to the OpenTinker/verl definition:

        observation_mask = attention_mask_response & ~response_mask
    """
    if observation_mask is not None:
        return observation_mask.float()

    response_length = response_mask.shape[1]
    attention_mask_response = attention_mask[:, -response_length:].float()
    return attention_mask_response * (1.0 - response_mask.float())


def compute_world_model_loss(
    log_prob: torch.Tensor,
    attention_mask: torch.Tensor,
    response_mask: torch.Tensor,
    observation_mask: Optional[torch.Tensor] = None,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Compute observation-token SFT loss used by the world-model objective."""
    resolved_observation_mask = compute_observation_mask(attention_mask=attention_mask,
                                                         response_mask=response_mask,
                                                         observation_mask=observation_mask)
    if not resolved_observation_mask.any().item():
        return None, resolved_observation_mask

    wm_sft_loss = -verl_F.masked_mean(log_prob, resolved_observation_mask)
    return wm_sft_loss, resolved_observation_mask
