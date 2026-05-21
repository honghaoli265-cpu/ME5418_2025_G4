#!/usr/bin/env python3
"""
Curriculum learning helper script.

Loads policy/value weights from a source checkpoint (e.g., None-mode training)
and continues PPO training in a harder environment mode (e.g., Random) without
restoring the saved environment state.
"""

import argparse
import os
from typing import Any, Dict

import numpy as np
import torch

from obstacle_env import ObstacleEnv
from learning_agent_ppo import PPOConfig, PPOTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curriculum warm-start training for the fencing agent.")
    parser.add_argument(
        "--init-checkpoint",
        required=True,
        help="Path to the source checkpoint whose policy/value weights will initialize training.",
    )
    parser.add_argument(
        "--model-type",
        choices=["lstm", "trans"],
        default="lstm",
        help="Backbone family used in the checkpoint/policy (default: lstm).",
    )
    parser.add_argument(
        "--target-obstacle-mode",
        choices=["static", "none", "random", "reactive", "alternate"],
        default="random",
        help="Environment obstacle mode for continued training (default: random).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of PPO updates to run for the target task (overrides checkpoint config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Training seed for the random-mode stage (overrides checkpoint config).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        help="Save checkpoints every N updates (overrides checkpoint config).",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        help="Root directory where new checkpoints will be stored (default: config/checkpoints).",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Optional explicit directory where checkpoints/metrics will be written.",
    )
    parser.add_argument(
        "--load-normalizers",
        action="store_true",
        help="Also copy running mean/std statistics from the source checkpoint.",
    )
    parser.add_argument(
        "--load-optimizers",
        action="store_true",
        help="Reuse optimizer states from the source checkpoint (off by default).",
    )
    return parser.parse_args()


def _load_checkpoint_payload(path: str) -> Dict[str, Any]:
    ckpt_path = os.path.abspath(path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    try:
        payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(ckpt_path, map_location="cpu")
    return payload


def _apply_saved_config(cfg: PPOConfig, saved: Dict[str, Any]) -> None:
    for key, value in saved.items():
        if not hasattr(cfg, key):
            continue
        setattr(cfg, key, value)


def _warm_start_from_checkpoint(
    trainer: PPOTrainer,
    checkpoint: Dict[str, Any],
    *,
    load_normalizers: bool = False,
    load_optimizers: bool = False,
) -> None:
    trainer.policy.load_state_dict(checkpoint["policy_state"])
    trainer.value.load_state_dict(checkpoint["value_state"])

    if load_optimizers:
        trainer.policy_optim.load_state_dict(checkpoint["policy_optimizer_state"])
        trainer.value_optim.load_state_dict(checkpoint["value_optimizer_state"])

    if load_normalizers:
        obs_rms = checkpoint.get("obs_rms")
        if obs_rms:
            np.copyto(trainer.obs_rms.mean, np.array(obs_rms.get("mean"), dtype=np.float64))
            np.copyto(trainer.obs_rms.var, np.array(obs_rms.get("var"), dtype=np.float64))
            trainer.obs_rms.count = float(obs_rms.get("count", trainer.obs_rms.count))
        reward_rms = checkpoint.get("reward_rms")
        if reward_rms:
            np.copyto(trainer.reward_rms.mean, np.array(reward_rms.get("mean"), dtype=np.float64))
            np.copyto(trainer.reward_rms.var, np.array(reward_rms.get("var"), dtype=np.float64))
            trainer.reward_rms.count = float(reward_rms.get("count", trainer.reward_rms.count))

    print(
        "[curriculum] Loaded policy/value parameters"
        f"{' and optimizer states' if load_optimizers else ''}"
        f"{' with RMS stats' if load_normalizers else ''}."
    )


def main() -> None:
    args = parse_args()

    checkpoint_payload = _load_checkpoint_payload(args.init_checkpoint)
    saved_cfg = checkpoint_payload.get("config") or {}

    config = PPOConfig()
    if isinstance(saved_cfg, dict):
        _apply_saved_config(config, saved_cfg)

    if args.iterations is not None:
        config.iterations = max(1, args.iterations)
    if args.seed is not None:
        config.seed = args.seed
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = max(1, args.checkpoint_interval)
    if args.checkpoint_root is not None:
        config.checkpoint_root = args.checkpoint_root

    if args.model_type == "lstm":
        from nerual_network_lstm import PolicyNetwork, ValueNetwork

        policy_kwargs = {"lstm_hidden_size": config.lstm_hidden_size}
        value_kwargs = {"lstm_hidden_size": config.lstm_hidden_size}
    else:
        from nerual_network_trans import PolicyNetwork, ValueNetwork

        policy_kwargs = {
            "seq_len": config.sequence_length,
            "embed_dim": config.transformer_embed_dim,
            "num_heads": config.transformer_num_heads,
            "num_layers": config.transformer_num_layers,
            "dropout": config.transformer_dropout,
        }
        value_kwargs = {
            "seq_len": config.sequence_length,
            "embed_dim": config.transformer_embed_dim,
            "num_heads": config.transformer_num_heads,
            "num_layers": config.transformer_num_layers,
            "dropout": config.transformer_dropout,
        }

    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(base_dir, "Fencing_agent&obstacle_description", "fencing_arm_ver3.xml")
    env = ObstacleEnv(xml_path=xml_path, obstacle_mode=args.target_obstacle_mode, render_mode=None)

    trainer = PPOTrainer(
        env,
        config,
        PolicyNetwork,
        ValueNetwork,
        policy_kwargs=policy_kwargs,
        value_kwargs=value_kwargs,
    )

    if args.checkpoint_dir:
        checkpoint_dir = os.path.abspath(args.checkpoint_dir)
        os.makedirs(checkpoint_dir, exist_ok=True)
        trainer.checkpoint_dir = checkpoint_dir
        trainer.checkpoint_root = os.path.dirname(checkpoint_dir) or trainer.checkpoint_root
        trainer.metrics_path = os.path.join(checkpoint_dir, "metrics.jsonl")
        trainer.run_timestamp = os.path.basename(checkpoint_dir.rstrip("/")) or trainer.run_timestamp

    _warm_start_from_checkpoint(
        trainer,
        checkpoint_payload,
        load_normalizers=args.load_normalizers,
        load_optimizers=args.load_optimizers,
    )

    trainer.train()


if __name__ == "__main__":
    main()
