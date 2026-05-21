# Group4 Multi-Agent Fencing Trainer

This project trains a pair of robotic fencing arms in a MuJoCo duel environment using Independent Proximal Policy Optimization (IPPO). The training loop supports recurrent LSTM or transformer policies, optional imitation-learning hooks, and periodic checkpointing so you can resume long experiments or analyze intermediate performance.

## Key Features
- **Custom DuelEnv** (`duel_env.py`): two 8-DoF arms simulated in MuJoCo with dense/shaped rewards for scoring, collisions, and weapon alignment.
- **Recurrent PPO agent** (`learning_agent_ippo.py` + `nerual_network_lstm.py`): sequence encoders feeding Gaussian policies/value heads, observation/reward normalization, and extensive tracking of episodic stats.
- **Training harness** (`main.py`): command-line interface to configure model type, seeds, checkpoints, and (future) GAIL hooks.
- **Visualization** (`main_test_duel_env.py`): run scripted bouts with the MuJoCo viewer for qualitative inspection.
- **Assets and checkpoints**: XML models under `Fencing_agent&obstacle_description/`, saved weights/metrics under `checkpoints/`.

## Repository Layout
- `duel_env.py` – Gymnasium-compatible environment that wraps MuJoCo physics, scoring logic, and reward shaping.
- `learning_agent_ippo.py` – PPO trainer, buffers, checkpoint/metric utilities, and dataclass config.
- `nerual_network_lstm.py` – LSTM policy/value definitions and helper MLP builder. (Add a transformer file if you use `--model-type trans`.)
- `main.py` – Entry point for training in the duel environment.
- `main_test_duel_env.py` – Simple script to spin up `DuelEnv` with deterministic actions for debugging or demo renders.
- `Fencing_agent&obstacle_description/` – MuJoCo XML model (`fencing_arm_ver3.xml`) plus meshes/textures referenced by the env.
- `checkpoints/` – Default output directory for saved weights, optimizer states, and `metrics.jsonl`.
- `*.log` – Example training logs captured via `tee`.

## Requirements
- Python 3.10+
- [MuJoCo](https://mujoco.org/) 2.3+ (ensure the shared library is discoverable via `LD_LIBRARY_PATH` or the official installer)
- `pip install` the following (or use conda equivalent):
  ```bash
  pip install torch gymnasium mujoco numpy
  ```
  Add any extras you use for logging or analysis (e.g., `pandas`, `matplotlib`).

## Installation
1. Clone or copy the project into your workspace.
2. (Optional) Create and activate a virtual environment/conda env.
3. Install the Python dependencies above.
4. Verify MuJoCo runs on your machine by launching `python -c "import mujoco"` and opening the viewer once (see MuJoCo docs).

## Training
Use `main.py` to launch PPO training:
```bash
python main.py \
  --model-type lstm \
  --iterations 4000 \
  --seed 888 \
  --checkpoint-root checkpoints/lstm/random/888 \
  --checkpoint-interval 20
```

Important arguments:
- `--model-type {lstm,trans}` – choose the policy/value backbone. LSTM weights live in `nerual_network_lstm.py`; make sure a transformer implementation exists if you select `trans`.
- `--iterations` – overrides the number of PPO update cycles (default `PPOConfig.iterations`).
- `--seed` – seeds MuJoCo, NumPy, and PyTorch for reproducibility.
- `--checkpoint-root` – directory where `learning_agent_ippo.py` will create timestamped folders containing checkpoints (`policy_*.pt`, `value_*.pt`) and `metrics.jsonl`.
- `--checkpoint-interval` – save cadence in PPO updates; set to `0` to disable.
- `--resume-checkpoint` – path to a previous checkpoint directory to continue training.
- `--use-gail` and `--expert-*` flags are stubbed in `PPOConfig` but currently raise `NotImplementedError` for DuelEnv; leave them at defaults.

Logs: redirect stdout/stderr with `tee` (examples in `*.log`) or integrate `tensorboard` inside `learning_agent_ippo.py`.

## Rendering / Debugging
To visually inspect the MuJoCo duel:
```bash
python main_test_duel_env.py
```
This script opens the viewer, runs a few scripted episodes, prints rewards, and closes cleanly. You can modify the hard-coded actions to replay policy outputs or collected trajectories.

## Checkpoints and Metrics
Each training run creates `checkpoints/<timestamp>/` containing:
- `policy_agent{idx}.pt` and `value_agent{idx}.pt`
- Optimizer states (`optim_*.pt`)
- `config.json` snapshot of `PPOConfig`
- `metrics.jsonl` per-update logs (episode returns, losses, success/failure counts)

Point `--resume-checkpoint` to one of these directories to restart a run. You can parse `metrics.jsonl` with your favorite plotting tool to track learning curves.

## Tips
- Start with shorter `--iterations` and `PPOConfig.rollout_steps` to validate setups before long jobs.
- If you hit MuJoCo viewer issues on headless servers, set `--render-mode none` (default) and use `mujoco-python-viewer` only on local machines with GPU/GUI access.
- Keep an eye on reward normalization (`RunningMeanStd`) when changing observation scales or reward shaping rules in `duel_env.py`.

Happy fencing! If you extend the trainer (e.g., add transformer networks, expert demonstrations, or richer logging), document the new switches in this README so future users can reproduce your experiments.
