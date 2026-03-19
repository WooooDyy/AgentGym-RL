"""World Model-Conditioned Entropy Regularized Clipping utilities.

This module adapts the WMC-ERC behavior used in OpenTinker to the
AgentGym-RL training loop. It uses per-token policy entropy as a world-model
uncertainty signal on observation tokens and applies a turn-level mask or
soft clipping coefficient to actor advantages.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


def compute_turn_boundaries(response_mask: torch.Tensor) -> List[List[Tuple[int, int]]]:
    """Extract contiguous assistant-token spans from response_mask.

    Each span corresponds to one action turn in the rollout response region.
    """
    boundaries_per_sample: List[List[Tuple[int, int]]] = []

    for sample_mask in response_mask.bool():
        sample_boundaries: List[Tuple[int, int]] = []
        start = None
        for idx, flag in enumerate(sample_mask.tolist()):
            if flag and start is None:
                start = idx
            elif not flag and start is not None:
                sample_boundaries.append((start, idx))
                start = None
        if start is not None:
            sample_boundaries.append((start, len(sample_mask)))
        boundaries_per_sample.append(sample_boundaries)

    return boundaries_per_sample


def compute_s_star(
    old_log_probs: torch.Tensor,
    entropys: torch.Tensor,
    response_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn policy-blind confidence scores."""
    s_star_per_sample: List[List[torch.Tensor]] = []

    for sample_idx, sample_turns in enumerate(turn_boundaries):
        sample_scores: List[torch.Tensor] = []
        for start, end in sample_turns:
            turn_mask = response_mask[sample_idx, start:end]
            if turn_mask.sum() <= 0:
                continue

            turn_log_probs = old_log_probs[sample_idx, start:end]
            turn_entropys = entropys[sample_idx, start:end]
            turn_probs = turn_log_probs.exp()
            turn_score = verl_F.masked_mean(turn_probs * (turn_entropys + turn_log_probs), turn_mask)
            sample_scores.append(turn_score)
        s_star_per_sample.append(sample_scores)

    return s_star_per_sample


def compute_h_wm(
    entropys: torch.Tensor,
    observation_mask: torch.Tensor,
    turn_boundaries: List[List[Tuple[int, int]]],
) -> List[List[torch.Tensor]]:
    """Compute per-turn world-model uncertainty from observation-token entropy.

    For each assistant span, we associate the following observation segment
    until the next assistant span or the end of valid response tokens.
    """
    h_wm_per_sample: List[List[torch.Tensor]] = []

    for sample_idx, sample_turns in enumerate(turn_boundaries):
        sample_scores: List[torch.Tensor] = []
        total_length = observation_mask.shape[1]

        for turn_idx, (_, end) in enumerate(sample_turns):
            next_start = sample_turns[turn_idx + 1][0] if turn_idx + 1 < len(sample_turns) else total_length
            env_start = min(end, total_length)
            env_end = min(next_start, total_length)

            if env_start >= env_end:
                sample_scores.append(torch.tensor(0.0, device=entropys.device))
                continue

            env_mask = observation_mask[sample_idx, env_start:env_end]
            if env_mask.sum() <= 0:
                sample_scores.append(torch.tensor(0.0, device=entropys.device))
                continue

            env_entropy = entropys[sample_idx, env_start:env_end]
            sample_scores.append(verl_F.masked_mean(env_entropy, env_mask))

        h_wm_per_sample.append(sample_scores)

    return h_wm_per_sample


def compute_dynamic_mask(
    s_star_per_sample: List[List[torch.Tensor]],
    h_wm_per_sample: List[List[torch.Tensor]],
    mu_base: float,
    mu_exp: float,
    eta_wm: float,
    lambda_wm: float,
    s_bar: float,
    sigma: float,
    clipping_method: str = "mask",
) -> List[List[float]]:
    """Compute per-turn dynamic entropy mask or soft clipping coefficient."""
    mask_per_sample: List[List[float]] = []

    for sample_idx in range(len(s_star_per_sample)):
        sample_masks: List[float] = []
        for turn_idx in range(len(s_star_per_sample[sample_idx])):
            s_t = s_star_per_sample[sample_idx][turn_idx].detach().item()
            h_t = h_wm_per_sample[sample_idx][turn_idx].detach().item()

            h_factor = eta_wm * np.exp(-lambda_wm * h_t)

            if s_t > s_bar:
                threshold = mu_base * h_factor * sigma
                diff = s_t - s_bar
            else:
                threshold = mu_exp * h_factor * sigma
                diff = s_bar - s_t

            if clipping_method == "mask":
                m_t = 1.0 if diff <= threshold else 0.0
            else:
                m_t = min(1.0, threshold / (diff + 1e-8))

            sample_masks.append(m_t)
        mask_per_sample.append(sample_masks)

    return mask_per_sample


def apply_wmc_erc(
    batch,
    entropys: torch.Tensor,
    wmc_erc_config,
    running_stats: Dict[str, float],
):
    """Apply WMC-ERC to actor advantages in-place."""
    enable = wmc_erc_config.get("enable", True) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "enable", True)
    if not enable:
        return batch, {}

    clipping_type = wmc_erc_config.get("clipping_type", "batch") if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "clipping_type", "batch")
    clipping_method = wmc_erc_config.get("clipping_method", "mask") if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "clipping_method", "mask")

    response_mask = batch.batch["response_mask"]
    old_log_probs = batch.batch["old_log_probs"]
    advantages = batch.batch["advantages"]

    response_length = advantages.shape[1]
    attention_mask_response = batch.batch["attention_mask"][:, -response_length:]
    observation_mask = attention_mask_response * (1.0 - response_mask)

    turn_boundaries = compute_turn_boundaries(response_mask)
    s_star = compute_s_star(old_log_probs, entropys, response_mask, turn_boundaries)
    h_wm = compute_h_wm(entropys, observation_mask, turn_boundaries)

    all_s = [score.item() for sample_scores in s_star for score in sample_scores]
    all_h = [score.item() for sample_scores in h_wm for score in sample_scores]
    if not all_s:
        return batch, {}

    batch_s_bar = np.mean(all_s)
    batch_s_std = np.std(all_s) + 1e-8
    batch_h_bar = np.mean(all_h) + 1e-8

    momentum = wmc_erc_config.get("momentum", 0.9) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "momentum", 0.9)
    if not running_stats.get("initialized", False):
        running_stats["s_bar"] = batch_s_bar
        running_stats["s_std"] = batch_s_std
        running_stats["h_bar"] = batch_h_bar
        running_stats["initialized"] = True
    else:
        running_stats["s_bar"] = (1 - momentum) * batch_s_bar + momentum * running_stats["s_bar"]
        running_stats["s_std"] = (1 - momentum) * batch_s_std + momentum * running_stats["s_std"]
        running_stats["h_bar"] = (1 - momentum) * batch_h_bar + momentum * running_stats["h_bar"]

    if clipping_type == "global":
        use_s_bar = running_stats["s_bar"]
        use_s_std = running_stats["s_std"]
    else:
        use_s_bar = batch_s_bar
        use_s_std = batch_s_std

    mu_base = float(wmc_erc_config.get("mu_base", 1.0) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "mu_base", 1.0))
    mu_exp = float(wmc_erc_config.get("mu_exp", 2.0) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "mu_exp", 2.0))
    eta_wm = float(wmc_erc_config.get("eta_wm", 1.0) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "eta_wm", 1.0))
    lambda_wm = float(wmc_erc_config.get("lambda_wm", 1.0) if hasattr(wmc_erc_config, "get") else getattr(wmc_erc_config, "lambda_wm", 1.0))

    mask = compute_dynamic_mask(
        s_star,
        h_wm,
        mu_base,
        mu_exp,
        eta_wm,
        lambda_wm,
        s_bar=use_s_bar,
        sigma=use_s_std,
        clipping_method=clipping_method,
    )

    for sample_idx, sample_turns in enumerate(turn_boundaries):
        for turn_idx, (start, end) in enumerate(sample_turns):
            advantages[sample_idx, start:end] *= mask[sample_idx][turn_idx]
    batch.batch["advantages"] = advantages

    all_m = [coef for sample_mask in mask for coef in sample_mask]
    num_collapsing_violated = 0
    num_exploration_violated = 0
    for sample_idx in range(len(s_star)):
        for turn_idx in range(len(s_star[sample_idx])):
            if mask[sample_idx][turn_idx] < 1.0:
                if s_star[sample_idx][turn_idx].item() > use_s_bar:
                    num_collapsing_violated += 1
                else:
                    num_exploration_violated += 1

    env_mask = observation_mask
    env_count = env_mask.sum()
    wm_nll = (-(old_log_probs * env_mask).sum() / (env_count + 1e-8)).item() if env_count > 0 else 0.0

    metrics = {
        "wmc_erc/batch_s_bar": float(batch_s_bar),
        "wmc_erc/batch_s_std": float(batch_s_std),
        "wmc_erc/batch_h_bar": float(batch_h_bar),
        "wmc_erc/running_s_bar": float(running_stats["s_bar"]),
        "wmc_erc/running_s_std": float(running_stats["s_std"]),
        "wmc_erc/running_h_bar": float(running_stats["h_bar"]),
        "wmc_erc/mask_ratio": float(np.mean(all_m)) if all_m else 1.0,
        "wmc_erc/num_violated_turns": sum(1 for coef in all_m if coef < 1.0),
        "wmc_erc/num_collapsing_violated": num_collapsing_violated,
        "wmc_erc/num_exploration_violated": num_exploration_violated,
        "wmc_erc/total_turns": len(all_m),
        "wmc_erc/wm_nll": wm_nll,
    }
    return batch, metrics
