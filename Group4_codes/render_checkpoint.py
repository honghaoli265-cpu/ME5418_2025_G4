#!/usr/bin/env python3
"""
Standalone rollout visualizer for trained checkpoints.

Usage example:
python render_checkpoint.py \
    --checkpoint /checkpoints/lstm/update_02000.pt \
    --model-type lstm \
    --obstacle-mode none
    --episodes 3 \
    --video-format mp4\
    --outpu-dir visual\
    --seed 128\
    # if don't use mp4, then --video-format gif
"""

import argparse
import os
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio
try:
    import imageio_ffmpeg as _imageio_ffmpeg  # type: ignore  # noqa: F401
except ImportError:
    _imageio_ffmpeg = None
import mujoco
import numpy as np
import torch

from obstacle_env import ObstacleEnv
from learning_agent_ppo import PPOConfig, RunningMeanStd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render PPO checkpoint rollouts to GIF/MP4.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint .pt file.")
    parser.add_argument(
        "--model-type",
        choices=["lstm", "trans"],
        default="lstm",
        help="Policy network type used during training.",
    )
    parser.add_argument(
        "--obstacle-mode",
        choices=["static", "none", "periodic", "reactive", "alternate","random"],
        default="periodic",
        help="Obstacle behavior to pass to the environment.",
    )
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record.")
    parser.add_argument("--max-steps", type=int, default=2000, help="Safety cap on steps per episode.")
    parser.add_argument("--seed", type=int, default=128, help="Base seed for env resets.")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS for the output video/GIF.")
    parser.add_argument(
        "--video-format",
        choices=["gif", "mp4"],
        default="mp4",
        help="Output container format.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="rollout_videos",
        help="Directory to store rendered rollouts.",
    )
    parser.add_argument("--width", type=int, default=640, help="Offscreen render width.")
    parser.add_argument("--height", type=int, default=480, help="Offscreen render height.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for policy inference.",
    )
    parser.add_argument(
        "--until-success",
        action="store_true",
        help="If set, ignore --max-steps and keep rolling until the env reports success/termination.",
    )
    return parser.parse_args()


def build_config(saved_cfg: Optional[Dict]) -> PPOConfig:
    cfg = PPOConfig()
    if isinstance(saved_cfg, dict):
        for key, value in saved_cfg.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
    return cfg


def make_policy(model_type: str, obs_dim: int, act_dim: int, cfg: PPOConfig) -> torch.nn.Module:
    if model_type == "lstm":
        from nerual_network_lstm import PolicyNetwork  # lazy import to avoid side effects

        policy = PolicyNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=cfg.hidden_dims,
            lstm_hidden_size=cfg.lstm_hidden_size,
        )
    else:
        from nerual_network_trans import PolicyNetwork

        policy = PolicyNetwork(
            obs_dim=obs_dim,
            act_dim=act_dim,
            hidden_dims=cfg.hidden_dims,
            seq_len=cfg.sequence_length,
            embed_dim=cfg.transformer_embed_dim,
            num_heads=cfg.transformer_num_heads,
            num_layers=cfg.transformer_num_layers,
            dropout=cfg.transformer_dropout,
        )
    return policy


def load_rms_state(obs_dim: int, state: Optional[Dict[str, np.ndarray]]) -> RunningMeanStd:
    rms = RunningMeanStd(shape=(obs_dim,))
    if not state:
        return rms
    if "mean" in state:
        np.copyto(rms.mean, np.array(state["mean"], dtype=np.float64))
    if "var" in state:
        np.copyto(rms.var, np.array(state["var"], dtype=np.float64))
    if "count" in state:
        rms.count = float(state["count"])
    return rms


def normalize_obs(obs: np.ndarray, rms: RunningMeanStd) -> np.ndarray:
    obs = obs.astype(np.float32)
    rms.update(obs[None, :])
    std = rms.std()
    std = np.where(std < 1e-6, 1.0, std)
    norm = (obs - rms.mean) / std
    return np.clip(norm, -10.0, 10.0).astype(np.float32)


def init_obs_window(norm_obs: np.ndarray, seq_len: int) -> Deque[np.ndarray]:
    return deque([norm_obs.copy() for _ in range(seq_len)], maxlen=seq_len)


def capture_frame(renderer: Optional[mujoco.Renderer], env: ObstacleEnv) -> Optional[np.ndarray]:
    if renderer is None:
        return None
    renderer.update_scene(env.data)
    frame = renderer.render()
    if frame is None:
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
    return frame


def rollout_episode(
    env: ObstacleEnv,
    policy: torch.nn.Module,
    rms: RunningMeanStd,
    seq_len: int,
    device: torch.device,
    renderer: Optional[mujoco.Renderer],
    seed: int,
    episode_idx: int,
    max_steps: int,
    fps: int,
    output_dir: str,
    video_format: str,
    until_success: bool,
) -> Tuple[int, float]:
    obs, _ = env.reset(seed=seed)
    obs = obs.astype(np.float32)
    norm_obs = normalize_obs(obs, rms)
    obs_window = init_obs_window(norm_obs, seq_len)
    frames: List[np.ndarray] = []
    first_frame = capture_frame(renderer, env)
    if first_frame is not None:
        frames.append(first_frame)

    total_reward = 0.0
    step_count = 0
    policy.eval()

    with torch.no_grad():
        while True:
            if not until_success and step_count >= max_steps:
                break
            obs_seq = np.stack(obs_window, axis=0)
            obs_tensor = torch.as_tensor(obs_seq, dtype=torch.float32, device=device).unsqueeze(0)
            action_tensor, _, _ = policy.sample(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, info = env.step(action)
            frame = info.get("frame")
            if frame is None:
                frame = capture_frame(renderer, env)
            if frame is not None:
                frames.append(frame)
            total_reward += float(reward)
            step_count += 1

            if step_count % 100 == 0:
                print(f"[debug] ep={episode_idx} step={step_count}")

            next_obs = next_obs.astype(np.float32)
            norm_next_obs = normalize_obs(next_obs, rms)
            obs_window.append(norm_next_obs)

            if terminated:
                break
            if truncated:
                break

    if frames:
        os.makedirs(output_dir, exist_ok=True)
        basename = f"rollout_ep{episode_idx:03d}"
        if video_format == "gif":
            path = os.path.join(output_dir, f"{basename}.gif")
            imageio.mimsave(path, frames, fps=fps)
        else:
            path = os.path.join(output_dir, f"{basename}.mp4")
            imageio.mimwrite(path, frames, fps=fps, quality=8, codec="libx264")
        print(f"[eval] Episode {episode_idx}: steps={step_count}, reward={total_reward:.2f}, saved to {path}")
    else:
        print(f"[eval] Episode {episode_idx}: steps={step_count}, reward={total_reward:.2f}, 无可用画面，跳过视频保存")
    return step_count, total_reward


def main() -> None:
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = build_config(checkpoint.get("config"))

    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(base_dir, "Fencing_agent&obstacle_description", "fencing_arm_ver3.xml")
    env = ObstacleEnv(xml_path=xml_path, obstacle_mode=args.obstacle_mode, render_mode="rgb_array")

    device = torch.device(args.device)
    policy = make_policy(args.model_type, env.observation_space.shape[0], env.action_space.shape[0], cfg)
    policy.load_state_dict(checkpoint["policy_state"])
    policy.to(device)
    policy.eval()

    rms = load_rms_state(env.observation_space.shape[0], checkpoint.get("obs_rms"))

    renderer = None
    try:
        renderer = mujoco.Renderer(env.model, width=args.width, height=args.height)
    except Exception as exc:
        print(f"[warn] 无法初始化 MuJoCo Renderer，将跳过视频保存: {exc}")

    total_steps = 0
    total_reward = 0.0
    for ep in range(args.episodes):
        steps, rew = rollout_episode(
            env=env,
            policy=policy,
            rms=rms,
            seq_len=cfg.sequence_length,
            device=device,
            renderer=renderer,
            seed=args.seed + ep,
            episode_idx=ep,
            max_steps=args.max_steps,
            fps=args.fps,
            output_dir=args.output_dir,
            video_format=args.video_format,
            until_success=args.until_success,
        )
        total_steps += steps
        total_reward += rew

    env.close()
    if hasattr(renderer, "close"):
        renderer.close()
    print(
        f"[eval] Finished {args.episodes} episodes | "
        f"avg_steps={total_steps / max(1, args.episodes):.1f} | "
        f"avg_reward={total_reward / max(1, args.episodes):.2f}"
    )


if __name__ == "__main__":
    main()
