import json
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

from duel_env import DuelEnv


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
    rollout_steps: int = 2048
    num_agents: int = 2
    agent_obs_dim: int = 24
    agent_act_dim: int = 8
    share_policy: bool = False
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

        self.num_agents = int(config.num_agents)
        self.agent_obs_dim = int(config.agent_obs_dim)
        self.agent_act_dim = int(config.agent_act_dim)
        total_obs_dim = env.observation_space.shape[0]
        total_act_dim = env.action_space.shape[0]
        expected_obs = self.num_agents * self.agent_obs_dim
        expected_act = self.num_agents * self.agent_act_dim
        if total_obs_dim != expected_obs:
            raise ValueError(
                f"Env observation dim {total_obs_dim} does not match num_agents*agent_obs_dim ({expected_obs})."
            )
        if total_act_dim != expected_act:
            raise ValueError(
                f"Env action dim {total_act_dim} does not match num_agents*agent_act_dim ({expected_act})."
            )
        if self.cfg.share_policy:
            raise NotImplementedError("Shared policies are not yet supported in the DuelEnv IPPO trainer.")
        if self.cfg.use_gail:
            raise NotImplementedError("GAIL is not wired up for the DuelEnv IPPO trainer.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policies = nn.ModuleList(
            [
                PolicyNetwork(self.agent_obs_dim, self.agent_act_dim, config.hidden_dims, **policy_kwargs).to(
                    self.device
                )
                for _ in range(self.num_agents)
            ]
        )
        self.values = nn.ModuleList(
            [
                ValueNetwork(self.agent_obs_dim, config.hidden_dims, **value_kwargs).to(self.device)
                for _ in range(self.num_agents)
            ]
        )
        self.policy_optims = [
            optim.Adam(policy.parameters(), lr=config.learning_rate) for policy in self.policies
        ]
        self.value_optims = [
            optim.Adam(value.parameters(), lr=config.learning_rate) for value in self.values
        ]

        # Online normalizers keep observation/reward scales in check / 在线归一化器用于稳定观测与奖励尺度
        self.obs_rms = [RunningMeanStd(shape=(self.agent_obs_dim,)) for _ in range(self.num_agents)]
        self.reward_rms = [RunningMeanStd(shape=(1,)) for _ in range(self.num_agents)]

        # Rollout buffers capture sequence inputs for RNN/Transformer / 采样缓存保存序列输入供循环或注意力网络使用
        self.obs_buf = np.zeros(
            (self.num_agents, config.rollout_steps, config.sequence_length, self.agent_obs_dim), dtype=np.float32
        )
        self.actions_buf = np.zeros((self.num_agents, config.rollout_steps, self.agent_act_dim), dtype=np.float32)
        self.rewards_buf = np.zeros((self.num_agents, config.rollout_steps), dtype=np.float32)
        self.raw_rewards_buf = np.zeros((self.num_agents, config.rollout_steps), dtype=np.float32)
        self.dones_buf = np.zeros(config.rollout_steps, dtype=np.float32)
        self.values_buf = np.zeros((self.num_agents, config.rollout_steps), dtype=np.float32)
        self.logprobs_buf = np.zeros((self.num_agents, config.rollout_steps), dtype=np.float32)
        self.running_ep_reward = np.zeros(self.num_agents, dtype=np.float32)
        self.completed_ep_rewards: List[List[float]] = [[] for _ in range(self.num_agents)]
        self.running_ep_reward_separate = np.zeros((self.num_agents, 6), dtype=np.float32)
        self.completed_ep_rewards_separate: List[List[np.ndarray]] = [[] for _ in range(self.num_agents)]
        self.completed_success: List[List[int]] = [[] for _ in range(self.num_agents)]
        self.completed_failure: List[List[int]] = [[] for _ in range(self.num_agents)]
        self.ep_num = 0

        self.obs_windows: List[Optional[Deque[np.ndarray]]] = [None for _ in range(self.num_agents)]
        self.last_obs: List[Optional[np.ndarray]] = [None for _ in range(self.num_agents)]
        self.last_raw_obs: List[Optional[np.ndarray]] = [None for _ in range(self.num_agents)]
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
    def _init_obs_window(self, agent_id: int, norm_obs: np.ndarray) -> None:
        norm_obs = norm_obs.astype(np.float32)
        self.obs_windows[agent_id] = deque(
            [norm_obs.copy() for _ in range(self.cfg.sequence_length)], maxlen=self.cfg.sequence_length
        )

    # Normalize raw observation with running stats / 利用运行统计量归一化原始观测
    def _normalize_obs(self, agent_id: int, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype(np.float32)
        self.obs_rms[agent_id].update(obs[None, :])
        norm = (obs - self.obs_rms[agent_id].mean) / self.obs_rms[agent_id].std()
        return np.clip(norm, -10.0, 10.0).astype(np.float32)

    # Normalize reward to stabilize critic updates / 归一化奖励以稳定价值网络
    def _normalize_reward(self, agent_id: int, reward: float) -> float:
        reward = float(reward)
        self.reward_rms[agent_id].update(np.array([[reward]], dtype=np.float32))
        scale = float(self.reward_rms[agent_id].std().squeeze())
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
    def _extract_env_obs(self) -> List[np.ndarray]:
        data = getattr(self.env, "data", None)
        if data is None:
            raise RuntimeError("Environment data buffer unavailable for observation extraction.")
        agent_qpos = np.array(data.qpos[:8], dtype=np.float32)
        agent_qvel = np.array(data.qvel[:8], dtype=np.float32)
        opponent_qpos = np.array(data.qpos[12:20], dtype=np.float32)
        obs_agent = np.concatenate([agent_qpos, agent_qvel, opponent_qpos]).astype(np.float32)

        opponent_qvel = np.array(data.qvel[12:20], dtype=np.float32)
        obs_opponent = np.concatenate([opponent_qpos, opponent_qvel, agent_qpos]).astype(np.float32)
        return [obs_agent, obs_opponent]

    # Persist trainer, optimizer, env, and RNG state / 保存训练器、优化器、环境及随机状态
    def save_checkpoint(self, update_idx: int, total_timesteps: int) -> str:
        obs_window: List[Optional[List[np.ndarray]]] = []
        for window in self.obs_windows:
            if window is None:
                obs_window.append(None)
            else:
                obs_window.append([np.array(item).copy() for item in window])

        checkpoint: Dict[str, Any] = {
            "update_idx": update_idx,
            "total_timesteps": total_timesteps,
            "config": asdict(self.cfg),
            "run_timestamp": self.run_timestamp,
            "policy_state": [policy.state_dict() for policy in self.policies],
            "value_state": [value.state_dict() for value in self.values],
            "policy_optimizer_state": [opt.state_dict() for opt in self.policy_optims],
            "value_optimizer_state": [opt.state_dict() for opt in self.value_optims],
            "obs_rms": [self._get_rms_state(rms) for rms in self.obs_rms],
            "reward_rms": [self._get_rms_state(rms) for rms in self.reward_rms],
            "running_ep_reward": self.running_ep_reward.copy(),
            "running_ep_reward_separate": self.running_ep_reward_separate.copy(),
            "completed_ep_rewards": [list(agent_rewards) for agent_rewards in self.completed_ep_rewards],
            "completed_ep_rewards_separate": [
                [np.array(arr).copy() for arr in agent_items] for agent_items in self.completed_ep_rewards_separate
            ],
            "completed_success": [list(agent_success) for agent_success in self.completed_success],
            "completed_failure": [list(agent_failure) for agent_failure in self.completed_failure],
            "ep_num": self.ep_num,
            "obs_window": obs_window,
            "last_obs": [
                None if obs is None else np.array(obs).copy() for obs in self.last_obs
            ],
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
                        # TODO：iterations 被强制设为“旧/新取最大”，无法通过命令行缩短继续训练的轮数；可以允许调用者在 load_checkpoint 之后显式覆写迭代上限，避免必须跑满历史值 (learning_agent_ippo.py (lines 297-301))。
                        self.cfg.iterations = max(int(value), int(self.cfg.iterations))
                    except Exception:
                        self.cfg.iterations = int(self.cfg.iterations)
                    continue
                setattr(self.cfg, key, value)

        policy_state = checkpoint.get("policy_state")
        if isinstance(policy_state, list):
            for agent_id, state in enumerate(policy_state):
                if agent_id < len(self.policies):
                    self.policies[agent_id].load_state_dict(state)
        elif policy_state is not None:
            for policy in self.policies:
                policy.load_state_dict(policy_state)

        value_state = checkpoint.get("value_state")
        if isinstance(value_state, list):
            for agent_id, state in enumerate(value_state):
                if agent_id < len(self.values):
                    self.values[agent_id].load_state_dict(state)
        elif value_state is not None:
            for value in self.values:
                value.load_state_dict(value_state)

        policy_opt_state = checkpoint.get("policy_optimizer_state")
        if isinstance(policy_opt_state, list):
            for agent_id, state in enumerate(policy_opt_state):
                if agent_id < len(self.policy_optims):
                    self.policy_optims[agent_id].load_state_dict(state)
        elif policy_opt_state is not None:
            for opt in self.policy_optims:
                opt.load_state_dict(policy_opt_state)

        value_opt_state = checkpoint.get("value_optimizer_state")
        if isinstance(value_opt_state, list):
            for agent_id, state in enumerate(value_opt_state):
                if agent_id < len(self.value_optims):
                    self.value_optims[agent_id].load_state_dict(state)
        elif value_opt_state is not None:
            for opt in self.value_optims:
                opt.load_state_dict(value_opt_state)

        obs_rms_state = checkpoint.get("obs_rms")
        if isinstance(obs_rms_state, list):
            for agent_id, state in enumerate(obs_rms_state):
                if agent_id >= len(self.obs_rms):
                    continue
                np.copyto(self.obs_rms[agent_id].mean, np.array(state.get("mean"), dtype=np.float64))
                np.copyto(self.obs_rms[agent_id].var, np.array(state.get("var"), dtype=np.float64))
                self.obs_rms[agent_id].count = float(state.get("count", self.obs_rms[agent_id].count))
        elif obs_rms_state:
            for rms in self.obs_rms:
                np.copyto(rms.mean, np.array(obs_rms_state.get("mean"), dtype=np.float64))
                np.copyto(rms.var, np.array(obs_rms_state.get("var"), dtype=np.float64))
                rms.count = float(obs_rms_state.get("count", rms.count))

        reward_rms_state = checkpoint.get("reward_rms")
        if isinstance(reward_rms_state, list):
            for agent_id, state in enumerate(reward_rms_state):
                if agent_id >= len(self.reward_rms):
                    continue
                np.copyto(self.reward_rms[agent_id].mean, np.array(state.get("mean"), dtype=np.float64))
                np.copyto(self.reward_rms[agent_id].var, np.array(state.get("var"), dtype=np.float64))
                self.reward_rms[agent_id].count = float(state.get("count", self.reward_rms[agent_id].count))
        elif reward_rms_state:
            for rms in self.reward_rms:
                np.copyto(rms.mean, np.array(reward_rms_state.get("mean"), dtype=np.float64))
                np.copyto(rms.var, np.array(reward_rms_state.get("var"), dtype=np.float64))
                rms.count = float(reward_rms_state.get("count", rms.count))

        running_reward = checkpoint.get("running_ep_reward", np.zeros(self.num_agents, dtype=np.float32))
        self.running_ep_reward = np.asarray(running_reward, dtype=np.float32).reshape(self.num_agents)
        running_sep = checkpoint.get(
            "running_ep_reward_separate", np.zeros((self.num_agents, 6), dtype=np.float32)
        )
        self.running_ep_reward_separate = np.asarray(running_sep, dtype=np.float32).reshape(self.num_agents, 6)

        completed_rewards = checkpoint.get("completed_ep_rewards", [[] for _ in range(self.num_agents)])
        self.completed_ep_rewards = [list(agent_rewards) for agent_rewards in completed_rewards]
        completed_sep = checkpoint.get(
            "completed_ep_rewards_separate", [[] for _ in range(self.num_agents)]
        )
        self.completed_ep_rewards_separate = [
            [np.array(arr, dtype=np.float32) for arr in agent_items] for agent_items in completed_sep
        ]
        completed_success = checkpoint.get("completed_success", [[] for _ in range(self.num_agents)])
        self.completed_success = [list(agent_list) for agent_list in completed_success]
        completed_failure = checkpoint.get("completed_failure", [[] for _ in range(self.num_agents)])
        self.completed_failure = [list(agent_list) for agent_list in completed_failure]
        while len(self.completed_ep_rewards) < self.num_agents:
            self.completed_ep_rewards.append([])
        while len(self.completed_ep_rewards_separate) < self.num_agents:
            self.completed_ep_rewards_separate.append([])
        while len(self.completed_success) < self.num_agents:
            self.completed_success.append([])
        while len(self.completed_failure) < self.num_agents:
            self.completed_failure.append([])
        self.ep_num = int(checkpoint.get("ep_num", 0))

        obs_window_state = checkpoint.get("obs_window")
        self.obs_windows = []
        if isinstance(obs_window_state, list):
            for window in obs_window_state:
                if window is None:
                    self.obs_windows.append(None)
                else:
                    self.obs_windows.append(
                        deque([np.array(item, dtype=np.float32) for item in window], maxlen=self.cfg.sequence_length)
                    )
        else:
            self.obs_windows = [None for _ in range(self.num_agents)]
        while len(self.obs_windows) < self.num_agents:
            self.obs_windows.append(None)

        last_obs_state = checkpoint.get("last_obs")
        if isinstance(last_obs_state, list):
            self.last_obs = [
                None if obs is None else np.array(obs, dtype=np.float32) for obs in last_obs_state
            ]
        elif last_obs_state is None:
            self.last_obs = [None for _ in range(self.num_agents)]
        else:
            obs_array = np.array(last_obs_state, dtype=np.float32)
            split_obs = np.split(obs_array, self.num_agents)
            self.last_obs = [arr.copy() for arr in split_obs]
        while len(self.last_obs) < self.num_agents:
            self.last_obs.append(None)

        self.last_done = bool(checkpoint.get("last_done", False))
        for agent_id in range(self.num_agents):
            if self.obs_windows[agent_id] is None and self.last_obs[agent_id] is not None:
                self._init_obs_window(agent_id, self.last_obs[agent_id].astype(np.float32))

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
            extracted_obs = self._extract_env_obs()
            self.last_raw_obs = [arr.copy() for arr in extracted_obs]
        except Exception:
            self.last_raw_obs = [None for _ in range(self.num_agents)]

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
    def collect_rollout(self, start_obs: List[np.ndarray]) -> Tuple[List[np.ndarray], bool]:
        obs = [arr.astype(np.float32) for arr in start_obs]
        for agent_id in range(self.num_agents):
            if self.obs_windows[agent_id] is None:
                self._init_obs_window(agent_id, obs[agent_id])
        raw_obs = []
        for agent_id in range(self.num_agents):
            if self.last_raw_obs[agent_id] is not None:
                raw_obs.append(self.last_raw_obs[agent_id].copy())
            else:
                raw_obs.append(np.zeros_like(obs[agent_id]))

        last_done = False
        for step in range(self.cfg.rollout_steps):
            actions = []
            log_probs = []
            values = []
            obs_seqs = []
            for agent_id in range(self.num_agents):
                obs_seq = np.stack(self.obs_windows[agent_id], axis=0)
                obs_tensor = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action_tensor, log_prob_tensor, _ = self.policies[agent_id].sample(obs_tensor)
                    value_tensor = self.values[agent_id](obs_tensor)
                action = action_tensor.squeeze(0).cpu().numpy()
                obs_seqs.append(obs_seq)
                actions.append(action)
                log_probs.append(float(log_prob_tensor.cpu().item()))
                values.append(float(value_tensor.cpu().item()))

            joint_action = np.concatenate(actions, axis=0)
            next_obs, reward, terminated, truncated, info = self.env.step(joint_action)
            last_terminated = bool(terminated)
            last_truncated = bool(truncated)
            last_done = last_terminated or last_truncated

            reward = np.asarray(reward, dtype=np.float32)
            if reward.shape[0] != self.num_agents:
                raise ValueError("Environment must return per-agent rewards for IPPO.")
            info_rew = info.get("R_separate")
            if info_rew is None:
                reward_terms = np.zeros((self.num_agents, 6), dtype=np.float32)
            else:
                reward_terms = np.asarray(info_rew, dtype=np.float32).reshape(self.num_agents, -1)

            split_next_obs = np.split(next_obs.astype(np.float32), self.num_agents)

            for agent_id in range(self.num_agents):
                self.obs_buf[agent_id, step] = obs_seqs[agent_id]
                self.actions_buf[agent_id, step] = actions[agent_id]
                self.raw_rewards_buf[agent_id, step] = reward[agent_id]
                norm_reward = self._normalize_reward(agent_id, reward[agent_id])
                self.rewards_buf[agent_id, step] = norm_reward
                self.values_buf[agent_id, step] = values[agent_id]
                self.logprobs_buf[agent_id, step] = log_probs[agent_id]
                self.running_ep_reward[agent_id] += reward[agent_id]
                self.running_ep_reward_separate[agent_id] = (
                    self.running_ep_reward_separate[agent_id] + reward_terms[agent_id]
                )

                norm_next_obs = self._normalize_obs(agent_id, split_next_obs[agent_id])
                self.obs_windows[agent_id].append(norm_next_obs)
                obs[agent_id] = norm_next_obs
                raw_obs[agent_id] = split_next_obs[agent_id].copy()
                self.last_raw_obs[agent_id] = split_next_obs[agent_id].copy()

            self.dones_buf[step] = float(last_done)

            if last_terminated:
                for agent_id in range(self.num_agents):
                    self.completed_ep_rewards[agent_id].append(float(self.running_ep_reward[agent_id]))
                    self.running_ep_reward[agent_id] = 0.0
                    agent_terms = self.running_ep_reward_separate[agent_id].copy()
                    if agent_terms[1] > 0.0:
                        self.completed_success[agent_id].append(1)
                        self.completed_failure[agent_id].append(0)
                    elif agent_terms[2] > 0.0:
                        self.completed_success[agent_id].append(0)
                        self.completed_failure[agent_id].append(1)
                    else:
                        self.completed_success[agent_id].append(0)
                        self.completed_failure[agent_id].append(0)
                    self.completed_ep_rewards_separate[agent_id].append(agent_terms)
                    self.running_ep_reward_separate[agent_id] = np.zeros(6, dtype=np.float32)
                self.ep_num += 1

            if last_done:
                reset_obs, _ = self.env.reset()
                reset_obs = reset_obs.astype(np.float32)
                split_reset = np.split(reset_obs, self.num_agents)
                for agent_id in range(self.num_agents):
                    self.last_raw_obs[agent_id] = split_reset[agent_id].copy()
                    norm_reset = self._normalize_obs(agent_id, split_reset[agent_id])
                    self._init_obs_window(agent_id, norm_reset)
                    obs[agent_id] = norm_reset
                    raw_obs[agent_id] = split_reset[agent_id].copy()

        return obs, last_done

    # Compute GAE(lambda) advantages and returns / 计算 GAE(lambda) 优势及回报
    def compute_gae(self, next_values: np.ndarray, last_done: bool) -> Tuple[np.ndarray, np.ndarray]:
        advantages = np.zeros_like(self.rewards_buf)
        returns = np.zeros_like(self.rewards_buf)
        for agent_id in range(self.num_agents):
            lastgaelam = 0.0
            for step in reversed(range(self.cfg.rollout_steps)):
                if step == self.cfg.rollout_steps - 1:
                    next_non_terminal = 1.0 - float(last_done)
                    next_value = next_values[agent_id]
                else:
                    next_non_terminal = 1.0 - self.dones_buf[step + 1]
                    next_value = self.values_buf[agent_id, step + 1]
                delta = (
                    self.rewards_buf[agent_id, step]
                    + self.cfg.gamma * next_value * next_non_terminal
                    - self.values_buf[agent_id, step]
                )
                lastgaelam = delta + self.cfg.gamma * self.cfg.gae_lambda * next_non_terminal * lastgaelam
                advantages[agent_id, step] = lastgaelam
            returns[agent_id] = advantages[agent_id] + self.values_buf[agent_id]
            adv = advantages[agent_id]
            advantages[agent_id] = (adv - adv.mean()) / (adv.std() + 1e-8)
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
        metrics: Dict[str, float] = {}
        batch_size = self.cfg.rollout_steps
        for agent_id in range(self.num_agents):
            inds = np.arange(batch_size)
            obs_tensor = torch.as_tensor(self.obs_buf[agent_id], dtype=torch.float32, device=self.device)
            actions_tensor = torch.as_tensor(self.actions_buf[agent_id], dtype=torch.float32, device=self.device)
            old_logprobs_tensor = torch.as_tensor(self.logprobs_buf[agent_id], dtype=torch.float32, device=self.device)
            advantages_tensor = torch.as_tensor(advantages[agent_id], dtype=torch.float32, device=self.device)
            returns_tensor = torch.as_tensor(returns[agent_id], dtype=torch.float32, device=self.device)

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
                    if mb_inds.size == 0:
                        continue

                    mb_obs = obs_tensor[mb_inds]
                    mb_actions = actions_tensor[mb_inds]
                    mb_old_logprobs = old_logprobs_tensor[mb_inds]
                    mb_advantages = advantages_tensor[mb_inds]
                    mb_returns = returns_tensor[mb_inds]

                    new_logprobs, entropy, log_std = self.policies[agent_id].evaluate(mb_obs, mb_actions)
                    ratio = torch.exp(new_logprobs.squeeze(-1) - mb_old_logprobs)
                    surrogate1 = ratio * mb_advantages
                    surrogate2 = (
                        torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef) * mb_advantages
                    )
                    policy_loss = -torch.min(surrogate1, surrogate2).mean()

                    value_estimates = self.values[agent_id](mb_obs).squeeze(-1)
                    value_loss = nn.functional.mse_loss(value_estimates, mb_returns)

                    loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy.mean()

                    self.policy_optims[agent_id].zero_grad()
                    self.value_optims[agent_id].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policies[agent_id].parameters(), self.cfg.max_grad_norm)
                    nn.utils.clip_grad_norm_(self.values[agent_id].parameters(), self.cfg.max_grad_norm)
                    self.policy_optims[agent_id].step()
                    self.value_optims[agent_id].step()
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

            if batches > 0:
                prefix = f"agent{agent_id}"
                for key, value in metric_sums.items():
                    metrics[f"{prefix}/{key}"] = value / batches
        return metrics
    # Main PPO loop: rollout -> GAE -> optimize -> log -> checkpoint / 主训练循环：采样 → GAE → 优化 → 打印 → checkpoint
    def train(self) -> None:
        if self.has_loaded_checkpoint:
            obs_list: List[np.ndarray] = []
            for agent_id in range(self.num_agents):
                last = self.last_obs[agent_id]
                if last is None:
                    dummy = np.zeros(self.agent_obs_dim, dtype=np.float32)
                    obs_list.append(dummy)
                    self._init_obs_window(agent_id, dummy)
                else:
                    obs_list.append(last.astype(np.float32))
                    if self.obs_windows[agent_id] is None:
                        self._init_obs_window(agent_id, last.astype(np.float32))
            if any(item is None for item in self.last_raw_obs):
                try:
                    extracted = self._extract_env_obs()
                    self.last_raw_obs = [arr.copy() for arr in extracted]
                except Exception:
                    self.last_raw_obs = [np.zeros(self.agent_obs_dim, dtype=np.float32) for _ in range(self.num_agents)]
            obs = obs_list
        else:
            raw_obs, _ = self.env.reset(seed=self.cfg.seed)
            raw_obs = raw_obs.astype(np.float32)
            split_raw = np.split(raw_obs, self.num_agents)
            obs = []
            for agent_id, agent_obs in enumerate(split_raw):
                norm_obs = self._normalize_obs(agent_id, agent_obs)
                self._init_obs_window(agent_id, norm_obs)
                obs.append(norm_obs)
                self.last_obs[agent_id] = norm_obs.copy()
                self.last_raw_obs[agent_id] = agent_obs.copy()
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
            self.last_obs = [arr.copy() for arr in obs]
            self.last_done = last_done

            next_values = np.zeros(self.num_agents, dtype=np.float32)
            with torch.no_grad():
                if not last_done:
                    for agent_id in range(self.num_agents):
                        obs_seq = np.stack(self.obs_windows[agent_id], axis=0)
                        obs_tensor = torch.as_tensor(obs_seq, dtype=torch.float32, device=self.device).unsqueeze(0)
                        next_val = self.values[agent_id](obs_tensor).cpu().numpy().squeeze(0)
                        next_values[agent_id] = float(next_val)

            advantages, returns = self.compute_gae(next_values, last_done)
            explained_variance = [
                self._explained_variance(returns[agent_id], self.values_buf[agent_id].copy())
                for agent_id in range(self.num_agents)
            ]
            train_metrics = self.update(advantages, returns) or {}
            for agent_id, ev in enumerate(explained_variance):
                train_metrics[f"agent{agent_id}/explained_variance"] = ev
            train_metrics["clip_range"] = float(self.cfg.clip_coef)
            train_metrics["learning_rate"] = float(self.cfg.learning_rate)
            train_metrics["n_updates"] = update

            time_elapsed = time.time() - start_time
            iteration_elapsed = time.time() - iteration_start_time
            fps = self.cfg.rollout_steps / max(iteration_elapsed, 1e-6)
            total_timesteps = update * self.cfg.rollout_steps

            ep_rew_mean = [
                float(np.mean(agent_rewards[-10:])) if agent_rewards else 0.0
                for agent_rewards in self.completed_ep_rewards
            ]
            success_rate = [
                float(np.mean(agent_success[-100:])) if agent_success else 0.0
                for agent_success in self.completed_success
            ]
            failure_rate = [
                float(np.mean(agent_failure[-100:])) if agent_failure else 0.0
                for agent_failure in self.completed_failure
            ]

            ep_rew_mean_separate: List[np.ndarray] = []
            for agent_items in self.completed_ep_rewards_separate:
                arr = np.array(agent_items, dtype=np.float32)
                if arr.size > 0:
                    ep_rew_mean_separate.append(np.mean(arr[-10:], axis=0))
                else:
                    ep_rew_mean_separate.append(np.zeros(6, dtype=np.float32))

            ep_len_mean = (update * self.cfg.rollout_steps / self.ep_num) if self.ep_num else 0.0

            metrics_payload: Dict[str, Any] = {
                "update": int(update),
                "total_timesteps": int(total_timesteps),
                "time_elapsed": float(time_elapsed),
                "fps": float(fps),
                "ep_len_mean": float(ep_len_mean),
            }
            for agent_id in range(self.num_agents):
                prefix = f"agent{agent_id}"
                metrics_payload[f"{prefix}/ep_rew_mean"] = ep_rew_mean[agent_id]
                metrics_payload[f"{prefix}/success_rate"] = success_rate[agent_id]
                metrics_payload[f"{prefix}/failure_rate"] = failure_rate[agent_id]
                metrics_payload[f"{prefix}/rew_terms"] = [
                    float(x) for x in ep_rew_mean_separate[agent_id].ravel().tolist()
                ]
            for key, value in train_metrics.items():
                if isinstance(value, (int, float)):
                    metrics_payload[key] = float(value)
            self._append_metrics(metrics_payload)

            print("----------------------------------------")
            print("rollout：")
            print(f"ep_num {self.ep_num} | ep_len_mean {ep_len_mean:.2f}")
            for agent_id in range(self.num_agents):
                terms = ep_rew_mean_separate[agent_id]
                print(
                    f"agent{agent_id} -> ep_rew_mean {ep_rew_mean[agent_id]:.2f} | "
                    f"success {success_rate[agent_id]*100:.2f}% | "
                    f"failure {failure_rate[agent_id]*100:.2f}% | "
                    f"dist {terms[0]:.2f} | success_term {terms[1]:.2f} | failure_term {terms[2]:.2f} | "
                    f"collision {terms[3]:.2f} | action {terms[4]:.2f} | step {terms[5]:.2f}"
                )
            print("----------------------------------------")
            print(
                f"time： fps {fps:.2f} | iterations {update}/{total_updates} | "
                f"time_elapsed {time_elapsed:.2f}s | total_timesteps {total_timesteps}"
            )
            if train_metrics:
                print("train：")
                for key, value in train_metrics.items():
                    print(f"{key}: {value:.4f}" if isinstance(value, float) else f"{key}: {value}")
            print("")

            if self.checkpoint_interval > 0 and update % self.checkpoint_interval == 0:
                self.save_checkpoint(update, total_timesteps)

        self.env.close()


def main() -> None:
    # Convenience entry for quick PPO runs / 方便在脚本模式下快速运行 PPO
    config = PPOConfig()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    xml_path = os.path.join(base_dir, "Fencing_agent&obstacle_description", "fencing_arm_ver3.xml")
    env = DuelEnv(xml_path=xml_path, render_mode=None)
    from nerual_network_lstm import PolicyNetwork, ValueNetwork

    policy_kwargs = {"lstm_hidden_size": config.lstm_hidden_size}
    value_kwargs = {"lstm_hidden_size": config.lstm_hidden_size}
    trainer = PPOTrainer(
        env,
        config,
        PolicyNetwork,
        ValueNetwork,
        policy_kwargs=policy_kwargs,
        value_kwargs=value_kwargs,
    )
    trainer.train()


if __name__ == "__main__":
    main()
