#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os

from duel_env import DuelEnv
from learning_agent_ippo import PPOTrainer, PPOConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the fencing agent with PPO.")
    parser.add_argument(
        "--model-type",
        choices=["lstm", "trans"],
        default="lstm",
        help="Policy/value network family to use (default: lstm).",
    )
    parser.add_argument(
        "--render-mode",
        choices=["human", "rgb_array", "none"],
        default="none",
        help="MuJoCo viewer mode for the dueling env (default: none).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        help="Number of PPO updates/epochs to run (default: config setting).",
    )
    # parser.add_argument(
    #     "--total-timesteps",
    #     type=int,
    #     help="Override PPO total training timesteps (default: 600000).",
    # )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed used for training (default: 42).",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        help="Save checkpoints every N PPO updates (default: config setting).",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=str,
        help="Root directory to store checkpoints (default: checkpoints).",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        help="Path to a checkpoint file to resume training from.",
    )
    parser.add_argument(
        "--use-gail",
        action="store_true",
        help="Enable Generative Adversarial Imitation Learning with expert demonstrations.",
    )
    parser.add_argument(
        "--expert-path",
        type=str,
        help="Path to expert dataset (CSV/NPZ) with obs+action+flags for GAIL.",
    )
    parser.add_argument(
        "--gail-batch-size",
        type=int,
        help="Batch size for each GAIL discriminator update (default: config setting).",
    )
    parser.add_argument(
        "--gail-iters",
        type=int,
        help="Number of discriminator steps per PPO update (default: config setting).",
    )
    parser.add_argument(
        "--gail-hidden-dims",
        type=int,
        nargs="+",
        help="Hidden layer sizes for the GAIL discriminator MLP.",
    )
    parser.add_argument(
        "--gail-learning-rate",
        type=float,
        help="Learning rate for the GAIL discriminator optimizer.",
    )
    parser.add_argument(
        "--gail-reward-scale",
        type=float,
        help="Scaling factor applied to the discriminator-based reward (default: 1.0).",
    )
    parser.add_argument(
        "--gail-mix-ratio",
        type=float,
        help="Ratio between GAIL reward and environment reward (1.0 = pure GAIL).",
    )
    parser.add_argument(
        "--gail-grad-penalty",
        type=float,
        help="Coefficient for gradient penalty regularization in the discriminator.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.model_type == "lstm":
        from nerual_network_lstm import PolicyNetwork, ValueNetwork
        model_id = "lstm"
    else:
        from nerual_network_trans import PolicyNetwork, ValueNetwork
        model_id = "trans"

    config = PPOConfig()

    total_timesteps = getattr(args, "total_timesteps", None)
    if total_timesteps is not None:
        config.total_timesteps = total_timesteps
    if args.iterations is not None:
        config.iterations = max(1, args.iterations)
    if args.seed is not None:
        config.seed = args.seed
    if args.checkpoint_interval is not None:
        config.checkpoint_interval = max(1, args.checkpoint_interval)
    if args.checkpoint_root is not None:
        config.checkpoint_root = args.checkpoint_root
    config.use_gail = args.use_gail
    if args.expert_path is not None:
        config.expert_data_path = args.expert_path
    if args.gail_batch_size is not None:
        config.gail_batch_size = max(1, args.gail_batch_size)
    if args.gail_iters is not None:
        config.gail_update_iters = max(1, args.gail_iters)
    if args.gail_hidden_dims:
        config.gail_hidden_dims = tuple(int(h) for h in args.gail_hidden_dims)
    if args.gail_learning_rate is not None:
        config.gail_learning_rate = float(args.gail_learning_rate)
    if args.gail_reward_scale is not None:
        config.gail_reward_scale = float(args.gail_reward_scale)
    if args.gail_mix_ratio is not None:
        config.gail_mix_ratio = float(args.gail_mix_ratio)
    if args.gail_grad_penalty is not None:
        config.gail_grad_penalty = float(args.gail_grad_penalty)

    if model_id == "lstm":
        policy_kwargs = {"lstm_hidden_size": config.lstm_hidden_size}
        value_kwargs = {"lstm_hidden_size": config.lstm_hidden_size}
    else:
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
    render_mode = None if args.render_mode == "none" else args.render_mode
    env = DuelEnv(xml_path=xml_path, render_mode=render_mode)
    trainer = PPOTrainer(
        env,
        config,
        PolicyNetwork,
        ValueNetwork,
        policy_kwargs=policy_kwargs,
        value_kwargs=value_kwargs,
    )
    if args.resume_checkpoint:
        trainer.load_checkpoint(args.resume_checkpoint)
    trainer.train()

if __name__ == "__main__":
    main()
