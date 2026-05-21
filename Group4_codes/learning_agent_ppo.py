import json
import math
import os
import time
from dataclasses import dataclass, asdict
from collections import deque
from typing import Any, Deque, Dict, List, Optional, Tuple

import mujoco
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from obstacle_env import ObstacleEnv
from gail_module import GAILModule


# Maintain online mean/variance trackers / 维护在线均值方差跟踪器
class RunningMeanStd:
    def __init__(self, shape):
        # Track running mean/variance for normalization / 跟踪归一化所需的滑动均值方差
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    # Update statistics with a new batch / 用新批次数据更新统计量
    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 0:
            x = x.reshape(1, 1)
        elif x.ndim == 1:
            x = x.reshape(1, -1)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    # Numerically stable Welford merge / 采用 Welford 算法稳定合并统计量
    def _update_from_moments(self, batch_mean, batch_var, batch_count) -> None:
        if batch_count == 0:
            return
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count
        self.mean = new_mean
        self.var = np.maximum(new_var, 1e-12)
        self.count = tot_count

    # Return current standard deviation / 返回当前标准差
    def std(self) -> np.ndarray:
        return np.sqrt(self.var + 1e-8)


# Centralized PPO hyper-parameter bundle / PPO 超参数集中配置
@dataclass
class PPOConfig:
    iterations: int = 500
    rollout_steps: int = 2048 # 2048
    minibatch_size: int = 256
    update_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    #0.2
    learning_rate: float = 3e-4
    #3e-4
    entropy_coef: float = 0.0
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    hidden_dims: Tuple[int, ...] = (256,)
    sequence_length: int = 10
    lstm_hidden_size: int = 256
    transformer_embed_dim: int = 128
    transformer_num_heads: int = 4
    transformer_num_layers: int = 2
    transformer_dropout: float = 0.1
    seed: int = 42
    checkpoint_interval: int = 10
    checkpoint_root: str = "checkpoints"
    # GAIL-specific settings /GAIL
    use_gail: bool = False
    expert_data_path: Optional[str] = None
    gail_batch_size: int = 256
    gail_update_iters: int = 5
    gail_hidden_dims: Tuple[int, ...] = (256, 256)
    gail_learning_rate: float = 3e-4
    gail_reward_scale: float = 1.0
    gail_mix_ratio: float = 1.0
    gail_grad_penalty: float = 0.0


# Full PPO training orchestrator / PPO 训练调度器
class PPOTrainer:
    # Wire env, networks, buffers, and (optional) GAIL / 连接环境、网络、缓存以及可选的 GAIL
    def __init__(
        self,
        env,
        config: PPOConfig,
        PolicyNetwork,
        ValueNetwork,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        value_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.env = env
        self.cfg = config
        policy_kwargs = policy_kwargs or {}
        value_kwargs = value_kwargs or {}

        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(obs_dim, act_dim, config.hidden_dims, **policy_kwargs).to(self.device)
        self.value = ValueNetwork(obs_dim, config.hidden_dims, **value_kwargs).to(self.device)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.value_optim = optim.Adam(self.value.parameters(), lr=config.learning_rate)

        # Online normalizers keep observation/reward scales in check / 在线归一化器用于稳定观测与奖励尺度
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.reward_rms = RunningMeanStd(shape=(1,))

        # Rollout buffers capture sequence inputs for RNN/Transformer / 采样缓存保存序列输入供循环或注意力网络使用
        self.obs_buf = np.zeros((config.rollout_steps, config.sequence_length, obs_dim), dtype=np.float32)
        self.actions_buf = np.zeros((config.rollout_steps, act_dim), dtype=np.float32)
        self.rewards_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        # 原始奖励缓冲
        self.raw_rewards_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.dones_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.values_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.logprobs_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.running_ep_reward = 0.0
        self.completed_ep_rewards = []
        self.running_ep_reward_separate = np.array([0, 0, 0, 0, 0, 0])
        self.completed_ep_rewards_separate = []
        self.completed_success = []
        self.completed_failure = []
        self.ep_num = 0

        self.obs_window: Optional[Deque[np.ndarray]] = None
        self.last_obs: Optional[np.ndarray] = None
        # 原始观测缓存
        self.last_raw_obs: Optional[np.ndarray] = None
        self.last_done: bool = False
        self.checkpoint_interval = max(0, config.checkpoint_interval)
        self.checkpoint_root = os.path.abspath(config.checkpoint_root)
        os.makedirs(self.checkpoint_root, exist_ok=True)
        self.run_timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.checkpoint_dir = os.path.join(self.checkpoint_root, self.run_timestamp)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.checkpoint_dir, "metrics.jsonl")
        self.last_checkpoint_path: Optional[str] = None
        self.start_update_idx: int = 0
        self.has_loaded_checkpoint: bool = False
        # GAIL模块实例
        self.gail_module: Optional[GAILModule] = None
        self.gail_policy_obs: List[np.ndarray] = []
        self.gail_policy_actions: List[np.ndarray] = []
        self.last_gail_metrics: Dict[str, float] = {}
        if self.cfg.use_gail:
            self._init_gail_module(obs_dim, act_dim)

    # Build the discriminator helper when imitation is enabled / 启用模仿学习时初始化判别器
    def _init_gail_module(self, obs_dim: int, act_dim: int) -> None:
        if not self.cfg.expert_data_path:
            raise ValueError("GAIL is enabled but expert_data_path is empty.")
        self.gail_module = GAILModule(
            obs_dim=obs_dim,
            act_dim=act_dim,
            device=self.device,
            dataset_path=self.cfg.expert_data_path,
            hidden_dims=self.cfg.gail_hidden_dims,
            batch_size=self.cfg.gail_batch_size,
            iters_per_update=self.cfg.gail_update_iters,
            learning_rate=self.cfg.gail_learning_rate,
            reward_scale=self.cfg.gail_reward_scale,
            grad_penalty_coef=self.cfg.gail_grad_penalty,
        )

    # Mix discriminator rewards back into PPO buffer / 将判别器奖励混入 PPO 缓冲区
    def _apply_gail_rewards(self) -> None:
        if not self.cfg.use_gail or self.gail_module is None:
            return
        if not self.gail_policy_obs:
            return

        policy_obs = np.asarray(self.gail_policy_obs, dtype=np.float32)
        policy_actions = np.asarray(self.gail_policy_actions, dtype=np.float32)
        gail_rewards, metrics = self.gail_module.update_and_reward(policy_obs, policy_actions)
        if gail_rewards.shape[0] != self.cfg.rollout_steps:
            raise ValueError(
                f"GAIL reward length {gail_rewards.shape[0]} does not match rollout steps {self.cfg.rollout_steps}."
            )

        mix = float(np.clip(self.cfg.gail_mix_ratio, 0.0, 1.0))
        mixed_rewards = mix * gail_rewards + (1.0 - mix) * self.raw_rewards_buf
        for idx, reward in enumerate(mixed_rewards):
            self.rewards_buf[idx] = self._normalize_reward(float(reward))

        reward_mean = float(np.mean(gail_rewards)) if gail_rewards.size > 0 else 0.0
        self.last_gail_metrics = {
            "expert_loss": metrics.get("expert_loss", 0.0),
            "policy_loss": metrics.get("policy_loss", 0.0),
            "grad_penalty": metrics.get("grad_penalty", 0.0),
            "reward_mean": reward_mean,
        }

        self.gail_policy_obs.clear()
        self.gail_policy_actions.clear()
        self.raw_rewards_buf.fill(0.0)

    def _append_metrics(self, payload: Dict[str, Any]) -> None:
        if not payload:
            return
        try:
            with open(self.metrics_path, "a", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False)
                file.write("\n")
        except OSError as exc:
            print(f"[metrics] Failed to write metrics: {exc}")

    # Initialize rolling observation window for sequence models / 为序列模型初始化滑动观测窗口
    def _init_obs_window(self, norm_obs: np.ndarray) -> None:
        norm_obs = norm_obs.astype(np.float32)
        self.obs_window = deque([norm_obs.copy() for _ in range(self.cfg.sequence_length)], maxlen=self.cfg.sequence_length)

    # Normalize raw observation with running stats / 利用运行统计量归一化原始观测
    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32)
        self.obs_rms.update(obs[None, :])
        norm = (obs - self.obs_rms.mean) / self.obs_rms.std()
        return np.clip(norm, -10.0, 10.0).astype(np.float32)

    # Normalize reward to stabilize critic updates / 归一化奖励以稳定价值网络
    def _normalize_reward(self, reward: float) -> float:
        reward = float(reward)
        self.reward_rms.update(np.array([[reward]], dtype=np.float32))
        scale = float(self.reward_rms.std().squeeze())
        if scale < 1e-6:
            return reward
        return reward / scale

    # Snapshot RMS buffers for checkpointing / 将均值方差缓冲保存到 checkpoint
    def _get_rms_state(self, rms: RunningMeanStd) -> Dict[str, Any]:
        return {
            "mean": rms.mean.copy(),
            "var": rms.var.copy(),
            "count": rms.count,
        }

    # Capture RNG states (torch/np/env) / 捕获随机数发生器状态（torch/np/环境）
    def _get_rng_state(self) -> Dict[str, Any]:
        rng_state: Dict[str, Any] = {
            "torch": torch.get_rng_state(),
            "numpy": np.random.get_state(),
        }
        if torch.cuda.is_available():
            rng_state["cuda"] = torch.cuda.get_rng_state_all()
        env_rng = getattr(self.env, "np_random", None)
        if env_rng is not None:
            if hasattr(env_rng, "bit_generator"):
                rng_state["env"] = env_rng.bit_generator.state
            elif hasattr(env_rng, "get_state"):
                rng_state["env"] = env_rng.get_state()
        return rng_state

    # Collect MuJoCo sim buffers needed to resume rollout / 收集恢复 rollout 所需的 MuJoCo 模拟状态
    def _get_env_state(self) -> Dict[str, Any]:
        state: Dict[str, Any] = {}
        data = getattr(self.env, "data", None)
        if data is not None:
            for attr in ("qpos", "qvel", "ctrl"):
                if hasattr(data, attr):
                    try:
                        state[attr] = np.array(getattr(data, attr)).copy()
                    except Exception:
                        pass
            if hasattr(data, "time"):
                state["time"] = float(data.time)
        state["episode_step"] = getattr(self.env, "episode_step", None)
        state["max_episode_steps"] = getattr(self.env, "max_episode_steps", None)
        state["obstacle_mode"] = getattr(self.env, "obstacle_mode", None)
        return state

    # Apply serialized MuJoCo state back to env / 将序列化的 MuJoCo 状态恢复到环境里
    def _apply_env_state(self, env_state: Optional[Dict[str, Any]]) -> None:
        if not env_state:
            return
        data = getattr(self.env, "data", None)
        if data is None:
            return
        for attr in ("qpos", "qvel", "ctrl"):
            if attr in env_state and hasattr(data, attr):
                try:
                    np.copyto(getattr(data, attr), np.array(env_state[attr], dtype=np.float64))
                except Exception:
                    pass
        if "time" in env_state:
            data.time = float(env_state["time"])
        if "episode_step" in env_state and hasattr(self.env, "episode_step"):
            try:
                self.env.episode_step = int(env_state["episode_step"])
            except Exception:
                pass
        if "max_episode_steps" in env_state and hasattr(self.env, "max_episode_steps"):
            try:
                self.env.max_episode_steps = int(env_state["max_episode_steps"])
            except Exception:
                pass
        if "obstacle_mode" in env_state and env_state["obstacle_mode"] is not None:
            try:
                self.env.obstacle_mode = env_state["obstacle_mode"]
            except Exception:
                pass
        try:
            mujoco.mj_forward(self.env.model, data)
        except Exception:
            pass
    
    # Read raw observation directly from MuJoCo buffers / 直接从 MuJoCo 缓冲中提取原始观测
    def _extract_env_obs(self) -> np.ndarray:
        data = getattr(self.env, "data", None)
        if data is None:
            raise RuntimeError("Environment data buffer unavailable for observation extraction.")
        qpos = np.array(data.qpos[:8], dtype=np.float32)
        qvel = np.array(data.qvel[:8], dtype=np.float32)
        obstacle = np.array(data.qpos[12:20], dtype=np.float32)
        return np.concatenate([qpos, qvel, obstacle]).astype(np.float32)

    # Persist trainer, optimizer, env, and RNG state / 保存训练器、优化器、环境及随机状态
    def save_checkpoint(self, update_idx: int, total_timesteps: int) -> str:
        if self.obs_window is not None:
            obs_window = [np.array(item).copy() for item in self.obs_window]
        else:
            obs_window = None

        checkpoint: Dict[str, Any] = {
            "update_idx": update_idx,
            "total_timesteps": total_timesteps,
            "config": asdict(self.cfg),
            "run_timestamp": self.run_timestamp,
            "policy_state": self.policy.state_dict(),
            "value_state": self.value.state_dict(),
            "policy_optimizer_state": self.policy_optim.state_dict(),
            "value_optimizer_state": self.value_optim.state_dict(),
            "obs_rms": self._get_rms_state(self.obs_rms),
            "reward_rms": self._get_rms_state(self.reward_rms),
            "running_ep_reward": self.running_ep_reward,
            "running_ep_reward_separate": self.running_ep_reward_separate.copy(),
            "completed_ep_rewards": list(self.completed_ep_rewards),
            "completed_ep_rewards_separate": [
                np.array(arr).copy() for arr in self.completed_ep_rewards_separate
            ],
            "completed_success": list(self.completed_success),
            "completed_failure": list(self.completed_failure),
            "ep_num": self.ep_num,
            "obs_window": obs_window,
            "last_obs": None if self.last_obs is None else np.array(self.last_obs).copy(),
            "last_done": self.last_done,
            "rng_state": self._get_rng_state(),
            "env_state": self._get_env_state(),
        }

        filename = f"update_{update_idx:05d}.pt"
        path = os.path.join(self.checkpoint_dir, filename)
        tmp_path = path + ".tmp"

        try:
            torch.save(checkpoint, tmp_path)
        except RuntimeError as exc:
            if "PytorchStreamWriter" in str(exc):
                torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
            else:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                raise

        os.replace(tmp_path, path)
        self.last_checkpoint_path = path
        print(f"[checkpoint] Saved update {update_idx} to {path}")
        return path

    # Restore everything needed to resume training seamlessly / 恢复继续训练所需的全部状态
    def load_checkpoint(self, path: str) -> None:
        checkpoint_path = os.path.abspath(path)
        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location="cpu")

        saved_cfg = checkpoint.get("config")
        if isinstance(saved_cfg, dict):
            for key, value in saved_cfg.items():
                if not hasattr(self.cfg, key):
                    continue
                if key == "iterations":
                    try:
                        # TODO：iterations 被强制设为“旧/新取最大”，无法通过命令行缩短继续训练的轮数；可以允许调用者在 load_checkpoint 之后显式覆写迭代上限，避免必须跑满历史值 (learning_agent_ppo.py (lines 297-301))。
                        self.cfg.iterations = max(int(value), int(self.cfg.iterations))
                    except Exception:
                        self.cfg.iterations = int(self.cfg.iterations)
                    continue
                setattr(self.cfg, key, value)

        self.policy.load_state_dict(checkpoint["policy_state"])
        self.value.load_state_dict(checkpoint["value_state"])
        self.policy_optim.load_state_dict(checkpoint["policy_optimizer_state"])
        self.value_optim.load_state_dict(checkpoint["value_optimizer_state"])

        obs_rms_state = checkpoint.get("obs_rms")
        if obs_rms_state:
            np.copyto(self.obs_rms.mean, np.array(obs_rms_state.get("mean"), dtype=np.float64))
            np.copyto(self.obs_rms.var, np.array(obs_rms_state.get("var"), dtype=np.float64))
            self.obs_rms.count = float(obs_rms_state.get("count", self.obs_rms.count))

        reward_rms_state = checkpoint.get("reward_rms")
        if reward_rms_state:
            np.copyto(self.reward_rms.mean, np.array(reward_rms_state.get("mean"), dtype=np.float64))
            np.copyto(self.reward_rms.var, np.array(reward_rms_state.get("var"), dtype=np.float64))
            self.reward_rms.count = float(reward_rms_state.get("count", self.reward_rms.count))

        self.running_ep_reward = float(checkpoint.get("running_ep_reward", 0.0))
        self.running_ep_reward_separate = np.array(
            checkpoint.get("running_ep_reward_separate", np.zeros(6, dtype=np.float32)),
            dtype=np.float32,
        )
        self.completed_ep_rewards = list(checkpoint.get("completed_ep_rewards", []))
        self.completed_ep_rewards_separate = [
            np.array(arr, dtype=np.float32)
            for arr in checkpoint.get("completed_ep_rewards_separate", [])
        ]
        self.completed_success = list(checkpoint.get("completed_success", []))
        self.completed_failure = list(checkpoint.get("completed_failure", []))
        self.ep_num = int(checkpoint.get("ep_num", 0))

        obs_window_state = checkpoint.get("obs_window")
        if obs_window_state is not None:
            self.obs_window = deque(
                [np.array(item, dtype=np.float32) for item in obs_window_state],
                maxlen=self.cfg.sequence_length,
            )
        else:
            self.obs_window = None

        last_obs_state = checkpoint.get("last_obs")
        self.last_obs = None if last_obs_state is None else np.array(last_obs_state, dtype=np.float32)
        self.last_done = bool(checkpoint.get("last_done", False))
        if self.obs_window is None and self.last_obs is not None:
            self._init_obs_window(self.last_obs.astype(np.float32))

        rng_state = checkpoint.get("rng_state", {})
        torch_state = rng_state.get("torch")
        if torch_state is not None:
            if not isinstance(torch_state, torch.ByteTensor):
                torch_state = torch.tensor(torch_state, dtype=torch.uint8)
            torch.set_rng_state(torch_state)
        numpy_state = rng_state.get("numpy")
        if numpy_state is not None:
            np.random.set_state(numpy_state)
        cuda_state = rng_state.get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            if isinstance(cuda_state, (list, tuple)):
                cuda_state = [
                    cs if isinstance(cs, torch.ByteTensor) else torch.tensor(cs, dtype=torch.uint8)
                    for cs in cuda_state
                ]
            elif not isinstance(cuda_state, torch.ByteTensor):
                cuda_state = torch.tensor(cuda_state, dtype=torch.uint8)
            torch.cuda.set_rng_state_all(cuda_state)
        env_state_rng = rng_state.get("env")
        env_rng = getattr(self.env, "np_random", None)
        if env_rng is not None and env_state_rng is not None:
            if hasattr(env_rng, "bit_generator"):
                env_rng.bit_generator.state = env_state_rng
            elif hasattr(env_rng, "set_state"):
                env_rng.set_state(env_state_rng)

        env_state = checkpoint.get("env_state")
        self._apply_env_state(env_state)
        try:
            self.last_raw_obs = self._extract_env_obs()
        except Exception:
            self.last_raw_obs = None

        self.start_update_idx = int(checkpoint.get("update_idx", 0))
        total_timesteps = checkpoint.get("total_timesteps")
        if total_timesteps is not None:
            try:
                self.start_update_idx = max(self.start_update_idx, int(total_timesteps // self.cfg.rollout_steps))
            except Exception:
                pass

        checkpoint_dir = os.path.dirname(checkpoint_path)
        parent_dir = os.path.dirname(checkpoint_dir)
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_root = parent_dir if parent_dir else self.checkpoint_root
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.metrics_path = os.path.join(self.checkpoint_dir, "metrics.jsonl")
        self.run_timestamp = checkpoint.get("run_timestamp", os.path.basename(self.checkpoint_dir))
        self.last_checkpoint_path = checkpoint_path
        self.has_loaded_checkpoint = True
        print(f"[checkpoint] Loaded state from {checkpoint_path}; resuming at update {self.start_update_idx}.")

    # Roll out one PPO batch and populate buffers / 采样一批 PPO 数据并写入缓存
    def collect_rollout(self, start_obs: np.ndarray) -> Tuple[np.ndarray, bool, bool, bool]:
        # Core sampling loop: build sequence, act, and push transitions / 采样主循环：构造序列、执行动作并存储转移
        obs = start_obs.astype(np.float32)
        if self.obs_window is None:
            self._init_obs_window(obs)
        raw_obs = self.last_raw_obs.copy() if self.last_raw_obs is not None else np.zeros_like(obs)
        if self.cfg.use_gail and self.gail_module is not None:
            self.gail_policy_obs = []
            self.gail_policy_actions = []
        last_done = False
        last_terminated = False
        last_truncated = False
        for step in range(self.cfg.rollout_steps):
            obs_seq = np.stack(self.obs_window, axis=0)
            obs_tensor = torch.as_tensor(
                obs_seq, dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            with torch.no_grad():
                action_tensor, log_prob_tensor, _ = self.policy.sample(obs_tensor)
                value_tensor = self.value(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()
            log_prob = float(log_prob_tensor.cpu().item())
            value = float(value_tensor.cpu().item())

            if self.cfg.use_gail and self.gail_module is not None:
                self.gail_policy_obs.append(raw_obs.copy())
                self.gail_policy_actions.append(action.copy())

            next_obs, reward, terminated, truncated, info = self.env.step(action)
            last_terminated = bool(terminated)
            last_truncated = bool(truncated)
            last_done = last_terminated or last_truncated

            self.obs_buf[step] = obs_seq
            self.actions_buf[step] = action
            self.raw_rewards_buf[step] = reward
            if not self.cfg.use_gail or self.gail_module is None:
                norm_reward = self._normalize_reward(reward)
                self.rewards_buf[step] = norm_reward
            self.dones_buf[step] = float(last_done)
            self.values_buf[step] = value
            self.logprobs_buf[step] = log_prob
            self.running_ep_reward += reward
            self.running_ep_reward_separate = self.running_ep_reward_separate + info["R_separate"]

            next_obs = next_obs.astype(np.float32)
            norm_next_obs = self._normalize_obs(next_obs)
            self.obs_window.append(norm_next_obs)
            obs = norm_next_obs
            raw_obs = next_obs.copy()
            self.last_raw_obs = raw_obs.copy()
            last_terminated_step = 0
            if last_terminated:
                # 只有双方得分导致的episode结束时才更新reward
                self.completed_ep_rewards.append(self.running_ep_reward)
                self.running_ep_reward = 0.0
                if self.running_ep_reward_separate[1]:
                    self.completed_success.append(1) 
                    self.completed_failure.append(0)
                elif self.running_ep_reward_separate[2]:
                    self.completed_failure.append(1)
                    self.completed_success.append(0) 
                self.completed_ep_rewards_separate.append(self.running_ep_reward_separate)
                self.running_ep_reward_separate = np.array([0, 0, 0, 0, 0, 0])
                self.ep_num += 1

            if last_done:
                obs, _ = self.env.reset()
                obs = obs.astype(np.float32)
                self.last_raw_obs = obs.copy()
                norm_reset_obs = self._normalize_obs(obs)
                self._init_obs_window(norm_reset_obs)
                obs = norm_reset_obs
                raw_obs = self.last_raw_obs.copy()

        return obs, last_done

    # Compute GAE(lambda) advantages and returns / 计算 GAE(lambda) 优势及回报
    def compute_gae(self, next_value: np.ndarray, last_done: bool) -> Tuple[np.ndarray, np.ndarray]:
        # Generalized Advantage Estimation / 广义优势估计
        advantages = np.zeros_like(self.rewards_buf)
        lastgaelam = 0.0
        for step in reversed(range(self.cfg.rollout_steps)):
            if step == self.cfg.rollout_steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_values = next_value
            else:
                next_non_terminal = 1.0 - self.dones_buf[step + 1]
                next_values = self.values_buf[step + 1]
            delta = (
                self.rewards_buf[step]
                + self.cfg.gamma * next_values * next_non_terminal
                - self.values_buf[step]
            )
            lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * lastgaelam
            advantages[step] = lastgaelam
        returns = advantages + self.values_buf
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    # Explained variance helper between targets and predictions / 计算目标与预测之间的解释方差
    @staticmethod
    def _explained_variance(targets: np.ndarray, preds: np.ndarray) -> float:
        var_y = np.var(targets)
        if var_y < 1e-8:
            return 0.0
        return float(1.0 - np.var(targets - preds) / (var_y + 1e-8))

    # Run clipped-PPO mini-batch updates / 执行裁剪 PPO 的小批量更新
    def update(self, advantages: np.ndarray, returns: np.ndarray) -> Dict[str, float]:
        obs_tensor = torch.as_tensor(self.obs_buf, dtype=torch.float32, device=self.device)
        actions_tensor = torch.as_tensor(self.actions_buf, dtype=torch.float32, device=self.device)
        old_logprobs_tensor = torch.as_tensor(self.logprobs_buf, dtype=torch.float32, device=self.device)
        advantages_tensor = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns_tensor = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        batch_size = self.cfg.rollout_steps
        inds = np.arange(batch_size)
        metric_sums = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "std": 0.0,
        }
        batches = 0
        for _ in range(self.cfg.update_epochs):
            np.random.shuffle(inds)
            for start in range(0, batch_size, self.cfg.minibatch_size):
                end = start + self.cfg.minibatch_size
                mb_inds = inds[start:end]

                mb_obs = obs_tensor[mb_inds]
                mb_actions = actions_tensor[mb_inds]
                mb_old_logprobs = old_logprobs_tensor[mb_inds]
                mb_advantages = advantages_tensor[mb_inds]
                mb_returns = returns_tensor[mb_inds]

                new_logprobs, entropy, log_std = self.policy.evaluate(mb_obs, mb_actions)
                ratio = torch.exp(new_logprobs.squeeze(-1) - mb_old_logprobs)
                surrogate1 = ratio * mb_advantages
                surrogate2 = torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * mb_advantages
                policy_loss = -torch.min(surrogate1, surrogate2).mean()

                value_estimates = self.value(mb_obs).squeeze(-1)
                value_loss = nn.functional.mse_loss(value_estimates, mb_returns)

                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy.mean()

                self.policy_optim.zero_grad()
                self.value_optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                nn.utils.clip_grad_norm_(self.value.parameters(), self.cfg.max_grad_norm)
                self.policy_optim.step()
                self.value_optim.step()
                batches += 1
                approx_kl = torch.mean(mb_old_logprobs - new_logprobs.squeeze(-1)).item()
                clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.cfg.clip_coef).float()).item()
                entropy_mean = entropy.mean().item()
                std_mean = torch.exp(log_std).mean().item()
                metric_sums["policy_loss"] += float(policy_loss.item())
                metric_sums["value_loss"] += float(value_loss.item())
                metric_sums["entropy"] += float(entropy_mean)
                metric_sums["approx_kl"] += float(approx_kl)
                metric_sums["clip_fraction"] += float(clip_fraction)
                metric_sums["std"] += float(std_mean)
        if batches == 0:
            return {}
        return {key: value / batches for key, value in metric_sums.items()}
    # Main PPO loop: rollout -> GAE -> optimize -> log -> checkpoint / 主训练循环：采样 → GAE → 优化 → 打印 → checkpoint
    def train(self) -> None:
        # Training loop: rollout → advantage/return → PPO update / 训练循环：采样 → 计算优势回报 → 执行 PPO 更新
        if self.has_loaded_checkpoint and self.last_obs is not None and self.obs_window is not None:
            obs = self.last_obs.astype(np.float32)
            if len(self.obs_window) != self.cfg.sequence_length:
                self._init_obs_window(obs)
            if self.last_raw_obs is None:
                try:
                    self.last_raw_obs = self._extract_env_obs()
                except Exception:
                    self.last_raw_obs = np.zeros_like(obs)
        else:
            raw_obs, _ = self.env.reset(seed=self.cfg.seed)
            raw_obs = raw_obs.astype(np.float32)
            norm_obs = self._normalize_obs(raw_obs)
            self._init_obs_window(norm_obs)
            obs = norm_obs
            self.last_obs = obs.copy()
            self.last_raw_obs = raw_obs.copy()
            self.last_done = False

        total_updates = self.cfg.iterations
        start_time = time.time()

        start_update = self.start_update_idx
        if start_update >= total_updates:
            print(f"[checkpoint] Requested iterations ({total_updates}) already completed; nothing to train.")
            return

        for update in range(start_update + 1, total_updates + 1):
            iteration_start_time = time.time()
            obs, last_done = self.collect_rollout(obs)
            self.last_obs = obs.copy()
            self.last_done = last_done
            if self.cfg.use_gail and self.gail_module is not None:
                self._apply_gail_rewards()

            with torch.no_grad():
                if last_done:
                    next_value = 0.0
                else:
                    obs_seq = np.stack(self.obs_window, axis=0)
                    obs_tensor = torch.as_tensor(
                        obs_seq, dtype=torch.float32, device=self.device
                    ).unsqueeze(0)
                    next_value = self.value(obs_tensor).cpu().numpy().squeeze(0)
            advantages, returns = self.compute_gae(next_value, last_done)
            explained_variance = self._explained_variance(returns, self.values_buf.copy())
            train_metrics = self.update(advantages, returns) or {}
            train_metrics["explained_variance"] = explained_variance
            train_metrics["clip_range"] = float(self.cfg.clip_coef)
            train_metrics["learning_rate"] = float(self.cfg.learning_rate)
            train_metrics["n_updates"] = update

            time_elapsed = time.time() - start_time
            iteration_elapsed = time.time() - iteration_start_time
            fps = self.cfg.rollout_steps / iteration_elapsed
            total_timesteps = update * self.cfg.rollout_steps

            ep_rew_mean = (
                np.mean(self.completed_ep_rewards[-10:]) if self.completed_ep_rewards else 0.0
            )
            success_rate = (
                np.mean(self.completed_success[-100:]) if self.completed_success else 0.0
            )
            failuer_rate = (
                np.mean(self.completed_failure[-100:]) if self.completed_failure else 0.0
            )

            arr = np.array(self.completed_ep_rewards_separate, dtype=np.float32)
            if arr.size > 0:
                ep_rew_mean_separate = np.mean(arr[-10:], axis=0)
            else:
                ep_rew_mean_separate = np.zeros(6, dtype=np.float32)

            if self.ep_num == 0:
                ep_len_mean = 0
            else:
                ep_len_mean = update * self.cfg.rollout_steps / self.ep_num

            metrics_payload: Dict[str, Any] = {
                "update": int(update),
                "total_timesteps": int(total_timesteps),
                "time_elapsed": float(time_elapsed),
                "fps": float(fps),
                "ep_rew_mean": float(ep_rew_mean),
                "success_rate": float(success_rate),
                "failure_rate": float(failuer_rate),
                "ep_len_mean": float(ep_len_mean),
                "ep_rew_mean_separate": [float(x) for x in np.asarray(ep_rew_mean_separate).ravel().tolist()],
            }
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_payload[key] = float(value)
            if self.cfg.use_gail and self.last_gail_metrics:
                metrics_payload["gail"] = {
                    key: float(val) for key, val in self.last_gail_metrics.items() if isinstance(val, (int, float))
                }
            self._append_metrics(metrics_payload)

            print(
                f"----------------------------------------\n"
                f"rollout：\n"
                f"ep_num {self.ep_num} | \n"
                f"ep_len_mean {ep_len_mean} | \n"
                f"ep_rew_mean {ep_rew_mean:.2f} | \n"
                f"success_rate {success_rate*100:.2f}%  | \n"
                f"failuer_rate {failuer_rate*100:.2f}% | \n"
                f"ep_rew_mean_dist: {ep_rew_mean_separate[0]:.2f} |\n"
                f"ep_rew_mean_success: {ep_rew_mean_separate[1]:.2f} |\n"
                f"ep_rew_mean_failure: {ep_rew_mean_separate[2]:.2f} |\n"
                f"ep_rew_mean_collision: {ep_rew_mean_separate[3]:.2f} |\n"
                f"ep_rew_mean_action: {ep_rew_mean_separate[4]:.2f} |\n"
                f"ep_rew_mean_step: {ep_rew_mean_separate[5]:.2f} |\n"
                f"----------------------------------------\n"
                f"time：\n"
                f"fps {fps:.2f} | \n"
                f"iterations {update}/{total_updates} | \n"
                f"time_elapsed {time_elapsed:.2f}s | \n"
                f"total_timesteps {total_timesteps} | \n"
                f"----------------------------------------\n"
                f"train："
            )

            if train_metrics:
                print(
                    f"train metrics .. \n"
                    f"policy_loss {train_metrics.get('policy_loss', 0.0):.4f} | \n"
                    f"value_loss {train_metrics.get('value_loss', 0.0):.4f} | \n"
                    f"entropy {train_metrics.get('entropy', 0.0):.4f} | \n"
                    f"approx_kl {train_metrics.get('approx_kl', 0.0):.4f} | \n"
                    f"clip_fraction {train_metrics.get('clip_fraction', 0.0):.4f} | \n"
                    f"std {train_metrics.get('std', 0.0):.4f} | \n"
                    f"explained_variance {train_metrics.get('explained_variance', 0.0):.4f} | "
                )
                print(
                    f"train meta .. \n"
                    f"learning_rate {train_metrics.get('learning_rate', 0.0):.6f} | \n"
                    f"clip_range {train_metrics.get('clip_range', 0.0):.3f} | \n"
                    f"n_updates {train_metrics.get('n_updates', 0)} | "
                )

            if self.cfg.use_gail and self.last_gail_metrics:
                print(
                    f"GAIL metrics .. \n"
                    f"expert_loss {self.last_gail_metrics.get('expert_loss', 0.0):.4f} | \n"
                    f"policy_loss {self.last_gail_metrics.get('policy_loss', 0.0):.4f} | \n"
                    f"reward_mean {self.last_gail_metrics.get('reward_mean', 0.0):.4f} | "
                    + (
                        f" | grad_penalty {self.last_gail_metrics.get('grad_penalty', 0.0):.4f}"
                        if self.cfg.gail_grad_penalty > 0.0
                        else ""
                    )
                )           
            print(f"")

            if self.checkpoint_interval > 0 and update % self.checkpoint_interval == 0:
                self.save_checkpoint(update, total_timesteps)

        self.env.close()


def main() -> None:
    # Convenience entry for quick PPO runs / 方便在脚本模式下快速运行 PPO
    config = PPOConfig()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(base_dir, "Fencing_agent&obstacle_description", "fencing_arm_ver3.xml")
    env = ObstacleEnv(xml_path=xml_path, obstacle_mode="periodic", render_mode=None)
    trainer = PPOTrainer(env, config)
    trainer.train()


if __name__ == "__main__":
    main()
