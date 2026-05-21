# Learning Agent Submission – MuJoCo Fencing PPO

This document covers the **learning agent** portion of the project; this README focuses exclusively on the PPO/GAIL training stack (`learning_agent_ppo.py`, `gail_module.py`, policy/value heads, and CLI tooling).

---

## 1. Prerequisites

- Linux + Anaconda/Miniconda. CUDA GPU is optional; PyTorch automatically falls back to CPU.
- The shared `environment.yml` (same as previous submissions) already contains Gymnasium, MuJoCo 3.x, PyTorch 2.8 (CUDA 12 build), numpy, pandas, etc.
- MuJoCo XML/mesh assets remain in `Fencing_agent&obstacle_description/` but no additional downloads are needed.

```bash
conda env create -f environment.yml        # first time
conda env update -n drl_mujoco -f environment.yml  # thereafter
conda activate drl_mujoco
```

---

## 2. Learning-Agent Source Layout

| Path | Role |
| --- | --- |
| `learning_agent_ppo.py` | Core PPO trainer (buffers, normalization, checkpoints, CLI integration). |
| `gail_module.py` | Optional Generative Adversarial Imitation Learning helper (dataset loader + discriminator). |
| `nerual_network_lstm.py` | LSTM-based policy/value heads used by the trainer. |
| `nerual_network_trans.py` | Transformer-based policy/value heads used by the trainer. |
| `main_train.py` | CLI entry; instantiates `ObstacleEnv`, selects network family, and launches `PPOTrainer`. |
| `main_curriculum_train.py` | Curriculum-learning wrapper that loads a source checkpoint (e.g., `none_lstm_cl_02000.pt`) and continues PPO in a new obstacle mode without restoring the old env state. |
| `main_validation.py` | Rollout/visualization utility that loads a saved checkpoint, builds the matching policy (LSTM or Transformer), and replays episodes in the MuJoCo viewer GUI for qualitative inspection; supports obstacle-mode switching, manual checkpoint selection, and repeat-until-success runs. |
| `plot_metrics.py` | Visualize `metrics.jsonl` logs, incl. GAIL fields (supports watch mode). |
| `traditional_method_to_get_expert_dataset` | Standalone tool (`main_get_expert_dataset.py` plus its helpers) for producing demonstrations offline. |

Everything else (environment assets, datasets) is unchanged from the Gym submission.

### Traditional Expert Dataset Generator

`traditional_method_to_get_expert_dataset/` hosts a standalone script for producing demonstrations offline:

- `main_get_expert_dataset.py`
  - Loads the MuJoCo model and launches an interactive viewer loop.
  - Plans end-effector routes in a discretized workspace via D\* Lite.
  - Refreshes the obstacle occupancy grid (`workspace_grid`) at fixed intervals and triggers replanning when obstacles change or block the current path (toggles defined near the top of the script).
  - Converts the waypoint sequence into joint targets for the first 8 controllable joints using `simple_ik` (see `ik_solver.py`) and tracks them with a PD controller.
  - Streams the execution (with automatic replanning) while logging every step to `expert_dataset_from_dstar_lite.csv`, including `qpos0-7`, `qvel0-7`, `qpos12-19`, `action0-7` (joint targets), and a `terminated` flag.

---

## 3. High-Level Algorithm Flow

1. `PPOTrainer` seeds RNGs, builds the chosen policy/value networks, and allocates rollout buffers sized `rollout_steps × sequence_length`.
2. During each update, `collect_rollout`:
   - Maintains a sliding observation window for sequence-aware policies.
   - Samples tanh-squashed Gaussian actions, recording values/log-probs.
   - Optionally caches trajectories for the GAIL discriminator.
   - Logs reward decomposition (`info["R_separate"]`) and success/failure counters.
3. `compute_gae` applies Generalized Advantage Estimation to the normalized rewards.
4. `update` runs clipped-PPO optimization across multiple minibatches with gradient clipping.
5. Every `checkpoint_interval` updates, `save_checkpoint` snapshots Torch/Numpy RNG states, MuJoCo buffers (`qpos/qvel/ctrl`), RMS normalizers, optimizer moments, and rollout statistics so that `load_checkpoint` can resume exactly.
6. If GAIL is enabled, the discriminator trains on-policy vs expert batches each update and produces reward shaping signals that mix with the environment reward (`gail_mix_ratio`).

---

## 4. CLI Usage (Learning Agent)

```bash
# With GAIL (turn on imitation learning)
python main_train.py --model-type lstm --obstacle-mode static --checkpoint-root checkpoints --checkpoint-interval 20 --use-gail --expert-path robot_states.csv

# Without GAIL (pure PPO)
python main_train.py --model-type lstm --obstacle-mode static --checkpoint-root checkpoints --checkpoint-interval 20

# Force an LSTM-only run (GAIL disabled) for 1,000 PPO updates
python main_train.py --model-type lstm --iterations 1000 --obstacle-mode static --checkpoint-root checkpoints --checkpoint-interval 20

# Curriculum continuation: warm start Random mode from the provided None-mode checkpoint
python main_curriculum_train.py \
  --init-checkpoint none_lstm_cl_02000.pt \
  --model-type lstm \
  --target-obstacle-mode random \
  --iterations 2000 \
  --seed 796 \
  --checkpoint-dir checkpoints/random_cl_seed796

# Rollout visualization: inspect a obstacle-mode: none checkpoint in the MuJoCo viewer until success
python main_validation.py \
  --checkpoint lstm_none.pt \
  --model-type lstm \
  --obstacle-mode periodic \
  --episodes 3 \
  --until-success

# Rollout visualization: inspect a obstacle-mode: alternate checkpoint in the MuJoCo viewer until success
python main_validation.py \
  --checkpoint lstm_alternate.pt \
  --model-type lstm \
  --obstacle-mode periodic \
  --episodes 3 \
  --until-success

# Rollout visualization: inspect a obstacle-mode: random checkpoint in the MuJoCo viewer until success
python main_validation.py \
  --checkpoint lstm_random.pt \
  --model-type lstm \
  --obstacle-mode periodic \
  --episodes 3 \
  --until-success
```

Key flags:

| Flag | Description |
| --- | --- |
| `--model-type {lstm,trans}` | Selects the policy/value head implementation. |
| `--obstacle-mode {static,periodic,reactive,none}` | Passed into the MuJoCo env. |
| `--iterations N` | Override how many PPO update epochs to run. |
| `--seed`, `--total-timesteps`, `--checkpoint-*` | Override `PPOConfig` defaults. |
| `--resume-checkpoint path.pt` | Restores a previous training state. |
| `--use-gail` + `--expert-path` | Turns on imitation learning (other `--gail-*` knobs fine-tune batch size, learning rate, reward mix, grad penalty, etc.). |
| `main_curriculum_train.py --init-checkpoint --target-obstacle-mode` | Reuses policy/value weights from a source `.pt` (default `none_lstm_cl_02000.pt`) and trains a fresh environment mode for curriculum learning; accepts the same PPO overrides plus `--checkpoint-dir` to pin the output folder. |

All unspecified flags fall back to `PPOConfig` in `learning_agent_ppo.py`.

---

## 5. Configuration Reference

`PPOConfig` fields (excerpt):

| Field | Default | Purpose |
| --- | --- | --- |
| `iterations` | 500 | Number of PPO updates (each uses `rollout_steps` transitions); can be overridden via `--iterations`. |
| `rollout_steps` | 2048 | Horizon per update, also expected length of GAIL rewards. |
| `sequence_length` | 10 | Temporal window fed into LSTM/Transformer policies. |
| `hidden_dims` | `(256,)` | Feature MLP width shared by policy/value heads. |
| `checkpoint_interval` | 10 | Save cadence (set 0 to disable). |
| `use_gail` / `gail_*` | Various | Configure dataset path, discriminator architecture, reward mixing. |

You can override any field by editing the dataclass or via CLI (when exposed in `main_train.py`).

---

## 6. GAIL Workflow

1. Prepare expert data (`.csv`, `.npy`, `.npz`). Columns must contain at least 24 obs dims + 8 actions; optional `start/terminated` flags help with weighting but are not required.
2. Launch training with `--use-gail --expert-path <file>`. Relevant knobs:
   - `--gail-batch-size`, `--gail-iters`, `--gail-hidden-dims`
   - `--gail-learning-rate`, `--gail-reward-scale`, `--gail-mix-ratio`
   - `--gail-grad-penalty` (activates WGAN-GP–style regularization).
3. During each PPO update, the discriminator learns to classify expert vs on-policy tuples and emits dense rewards that are normalized + blended into the PPO buffer.
4. Metrics (`expert_loss`, `policy_loss`, `grad_penalty`, `reward_mean`) appear in the console after each update for quick health checks.

---

## 7. Debugging & Verification


- **Console logs (`learning_agent_ppo.py`)**: Each update prints:
  - Rolling episode counts, mean return, success/failure rate (last 100 episodes).
  - Reward decomposition averages for distance/success/failure/collision/action/step terms.
  - FPS, elapsed time, total timesteps processed, and iteration counters.
  - Optional GAIL metrics block.
- **Checkpoints**: Re-run with `--resume-checkpoint` to confirm determinism and mid-run recovery.

Example metric plotting command (`metrics.jsonl` is created per training run):

```bash
# Running example 
python plot_metrics.py --metrics-path checkpoints/20251107-202114/metrics.jsonl --fields policy_loss value_loss ep_rew_mean success_rate gail.expert_loss gail.policy_loss

# General situation
python plot_metrics.py \
  --metrics-path checkpoints/<timestamp>/metrics.jsonl \
  --fields policy_loss value_loss ep_rew_mean success_rate gail.expert_loss gail.policy_loss \
  --watch --watch-interval 3 # if update plot in real time
```

---

## 8. Checkpointing & Resume

- `learning_agent_ppo.py` writes checkpoints under `<checkpoint_root>/<timestamp>/update_XXXXX.pt`.
- Each file stores:
  - Policy/value weights + optimizer states.
  - Observation/reward running statistics.
  - Episode counters, reward history, last observation window.
  - RNG states for PyTorch CPU/GPU, NumPy, and the Gym environment.
  - MuJoCo `qpos/qvel/ctrl/time` so the simulation can keep stepping deterministically.
- Resume via `python main_train.py --resume-checkpoint checkpoints/<running-timestamp>/update_00050.pt` (you can still change future `iterations`, checkpoint cadence, or GAIL knobs afterward).

---

## 9. Next Steps

The learning agent stack delivered here is ready for integration with the already-submitted OpenAI Gym environment and neural-network deliverables. Future enhancements (outside this submission) could include:
- Persisting full training metrics to disk for visualization.
- Adding curriculum schedulers for `obstacle_mode`.
- Auto-exporting TorchScript checkpoints for deployment.

For the purposes of the current milestone, please evaluate the PPO/GAIL trainer using the commands and documentation above.
