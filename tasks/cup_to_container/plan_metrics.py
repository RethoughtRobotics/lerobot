"""Held-out plans measured in millimetres, as the robot runs them.

The noise-prediction loss scores a denoiser step, while the robot executes sampled plans. At run time a plan
arrives every 4 steps and lasts 24, so each executed step is the temporal ensemble (Zhao et al., ACT, 2023) of up
to 6 plans. Six samples per frame therefore allow both a single plan and that average to be scored.
"""

import torch


def plan_stats(plans: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    """Position statistics of K sampled plans against the demonstrated chunk.

    Args:
        plans: unnormalized plans (B, K, T, D), with position in metres in the first three dims.
        target: unnormalized demonstrated chunk (B, T, D).

    Returns:
        Means over frames and steps, in mm:
        plan_error_mm: distance from one sampled plan to the demonstration.
        merged_plan_error_mm: distance from the average of the K plans to the demonstration.
        plan_spread_mm: distance from one sampled plan to the average of the K plans.
    """
    merged = plans.mean(1)
    return {
        "plan_error_mm": 1000 * (plans - target[:, None])[..., :3].norm(dim=-1).mean().item(),
        "merged_plan_error_mm": 1000 * (merged - target)[..., :3].norm(dim=-1).mean().item(),
        "plan_spread_mm": 1000 * (plans - merged[:, None])[..., :3].norm(dim=-1).mean().item(),
    }


def sample_plans(policy, batch: dict, samples: int, stride: int, postprocessor, device, seed: int) -> torch.Tensor:
    """Sample plans for every `stride`-th frame of a preprocessed batch, unnormalized, (B, K, T, D), on the CPU.

    The RNG is forked so that sampling leaves the noise draws of the loss unchanged, and seeded so that the curve
    is comparable across steps.
    """
    config = policy.config
    start = config.n_obs_steps - 1
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
        torch.manual_seed(seed)
        conditioning = policy.observation_encoder.encode(policy._prepare_batch(batch))[::stride]
        repeated = conditioning.repeat_interleave(samples, 0)
        actions = policy.objective.conditional_sample(policy.noise_predictor, len(repeated), repeated)
    actions = actions[:, start : start + config.n_action_steps].float()
    return postprocessor(actions).cpu().view(-1, samples, *actions.shape[1:])
