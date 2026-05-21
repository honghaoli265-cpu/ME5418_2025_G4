#!/usr/bin/env python3
"""
Record rollouts of the traditional D* Lite + IK pipeline into GIF/MP4 files.

Example usage:
python render_checkpoint.py \
    --episodes 3 \
    --output-dir recordings \
    --video-format mp4 \
    --fps 30 \
    --obstacle-mode alternate
"""

import argparse
import os
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import imageio

try:
    import imageio_ffmpeg as _imageio_ffmpeg  # type: ignore  # noqa: F401
except ImportError:
    _imageio_ffmpeg = None
import mujoco
import mujoco.viewer
import numpy as np

from dstar_lite import DStarLite
from ik_solver import simple_ik

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
XML_PATH = os.path.join(PARENT_DIR, "Fencing_agent&obstacle_description", "fencing_arm_ver3.xml")

GRID_SIZE = (100, 20, 20)
EE_MIN = np.array([-4.0, -0.5, -0.5], dtype=np.float64)
EE_MAX = np.array([1.0, 0.5, 0.5], dtype=np.float64)
OBS_BODY_LIST = [f"link{i}_obstacle" for i in range(8)]

Kp = 50.0
Kd = 1.5
CONTROLLABLE_JOINTS = 8
STEP_PER_WAYPOINT = 200
OBSTACLE_JOINT_SLICE = range(12, 20)
OBSTACLE_ACTUATOR_IDS = np.arange(8, 16)

DEFAULT_STATIC_ACTION = np.array([0.0, -0.9, 0.6, -0.5, -0.5, 0.5, 0.0, 0.0], dtype=np.float32)
DEFAULT_ALTERNATE_ACTION1 = np.array([0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_ALTERNATE_ACTION2 = np.array([0.0, 0.9, 0.6, -0.7, -0.7, 0.5, 0.3, 0.3], dtype=np.float32)


@dataclass
class ObstacleControllerState:
    """Keeps per-episode state for obstacle motion policies."""

    random_action: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record the traditional planner rollout as a video.")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes to record.")
    parser.add_argument("--max-steps", type=int, default=20_000, help="Safety cap on steps per episode.")
    parser.add_argument("--seed", type=int, default=128, help="Seed for goal sampling/random obstacle motions.")
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS for the output video.")
    parser.add_argument("--width", type=int, default=640, help="Offscreen render width.")
    parser.add_argument("--height", type=int, default=480, help="Offscreen render height.")
    parser.add_argument(
        "--video-format",
        choices=["gif", "mp4"],
        default="mp4",
        help="Output container format.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="traditional_rollouts",
        help="Directory to store rendered rollouts.",
    )
    parser.add_argument(
        "--obstacle-mode",
        choices=["static", "none", "random", "reactive", "alternate"],
        default="alternate",
        help="Obstacle controller used during the rollout.",
    )
    parser.add_argument(
        "--alternate-steps",
        type=int,
        default=1000,
        help="How many sim steps to keep each alternate action.",
    )
    parser.add_argument(
        "--random-steps",
        type=int,
        default=1000,
        help="How often (in sim steps) to resample random obstacle actions.",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Also open an interactive MuJoCo viewer window while recording.",
    )
    return parser.parse_args()


def sample_random_goal() -> np.ndarray:
    x = np.random.uniform(-3.5, -2.5)
    y = np.random.uniform(-0.1, 0.1)
    z = np.random.uniform(-0.2, 0.2)
    return np.array([x, y, z], dtype=np.float64)


def ee2grid(ee_pos: np.ndarray) -> Tuple[int, int, int]:
    grid_pos = (ee_pos - EE_MIN) / (EE_MAX - EE_MIN) * (np.array(GRID_SIZE) - 1)
    clip = np.clip(grid_pos.astype(int), 0, np.array(GRID_SIZE) - 1)
    return int(clip[0]), int(clip[1]), int(clip[2])


def grid2ee(grid_pos: Sequence[int]) -> np.ndarray:
    frac = np.array(grid_pos) / (np.array(GRID_SIZE) - 1)
    return frac * (EE_MAX - EE_MIN) + EE_MIN


def fill_workspace_from_model(model: mujoco.MjModel, data: mujoco.MjData, workspace_grid: np.ndarray) -> np.ndarray:
    workspace_grid.fill(0)
    nx, ny, nz = GRID_SIZE
    for name in OBS_BODY_LIST:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id == -1:
            continue
        geom_start = model.body_geomadr[body_id]
        geom_count = model.body_geomnum[body_id]
        for j in range(geom_start, geom_start + geom_count):
            geom_pos = data.geom_xpos[j]
            geom_size = model.geom_size[j]
            lower = geom_pos - geom_size
            upper = geom_pos + geom_size
            grid_min = np.clip(
                ((lower - EE_MIN) / (EE_MAX - EE_MIN) * (np.array(GRID_SIZE) - 1)).astype(int),
                0,
                np.array(GRID_SIZE) - 1,
            )
            grid_max = np.clip(
                ((upper - EE_MIN) / (EE_MAX - EE_MIN) * (np.array(GRID_SIZE) - 1)).astype(int),
                0,
                np.array(GRID_SIZE) - 1,
            )
            workspace_grid[
                grid_min[0] : min(nx, grid_max[0] + 1),
                grid_min[1] : min(ny, grid_max[1] + 1),
                grid_min[2] : min(nz, grid_max[2] + 1),
            ] = 1
    return workspace_grid


def action_to_qpos_general(model: mujoco.MjModel, action: np.ndarray, joint_slice: Sequence[int]) -> np.ndarray:
    joint_indices = np.array(list(joint_slice))
    joint_range = model.jnt_range[joint_indices]
    q_min = joint_range[:, 0]
    q_max = joint_range[:, 1]
    return q_min + (np.clip(action, -1.0, 1.0) + 1.0) * 0.5 * (q_max - q_min)


def plan_path_and_ik(
    model: mujoco.MjModel,
    data_plan: mujoco.MjData,
    start_ee: np.ndarray,
    goal_ee: np.ndarray,
    workspace_grid: np.ndarray,
) -> Tuple[Optional[List[Tuple[int, int, int]]], Optional[np.ndarray], List[Optional[np.ndarray]]]:
    start_node = ee2grid(start_ee)
    goal_node = ee2grid(goal_ee)
    dstar = DStarLite(start_node, goal_node, workspace_grid.copy())
    dstar.compute_shortest_path()
    path_nodes = dstar.get_path()
    if not path_nodes:
        return None, None, []

    ee_path = np.array([grid2ee(node) for node in path_nodes])

    mujoco.mj_forward(model, data_plan)
    tip_name = "tip8_agent"
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_name)
    target_ori = data_plan.site_xmat[tip_id].reshape(3, 3)

    ee_qpos_path: List[Optional[np.ndarray]] = []
    for pos in ee_path:
        q = simple_ik(
            model,
            data_plan,
            np.array(pos, dtype=np.float64),
            target_ori=target_ori,
            controllable_joints=CONTROLLABLE_JOINTS,
        )
        ee_qpos_path.append(None if q is None else q.copy())
    return path_nodes, ee_path, ee_qpos_path


def check_collision_point(model: mujoco.MjModel, data: mujoco.MjData) -> int:
    for i in range(data.ncon):
        contact = data.contact[i]
        geom1_id = contact.geom1
        geom2_id = contact.geom2
        contype1 = model.geom_contype[geom1_id]
        contype2 = model.geom_contype[geom2_id]
        body1_id = model.geom_bodyid[geom1_id]
        body2_id = model.geom_bodyid[geom2_id]
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

        cond1 = (contype1 == 4 and "agent" in name1 and contype2 == 17 and "obstacle" in name2)
        cond2 = (contype2 == 4 and "agent" in name2 and contype1 == 17 and "obstacle" in name1)
        cond3 = (contype1 == 8 and "obstacle" in name1 and contype2 == 17 and "agent" in name2)
        cond4 = (contype2 == 8 and "obstacle" in name2 and contype1 == 17 and "agent" in name1)

        if (cond1 or cond2) and (cond3 or cond4):
            return 3
        if cond1 or cond2:
            return 2
        if cond3 or cond4:
            return 1
    return 0


def capture_frame(renderer: Optional[mujoco.Renderer], data: mujoco.MjData) -> Optional[np.ndarray]:
    if renderer is None:
        return None
    renderer.update_scene(data)
    frame = renderer.render()
    if frame is None:
        return None
    if frame.dtype != np.uint8:
        frame = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
    return frame


def save_video(
    frames: List[np.ndarray],
    output_dir: str,
    episode_idx: int,
    video_format: str,
    fps: int,
) -> Optional[str]:
    if not frames:
        return None
    os.makedirs(output_dir, exist_ok=True)
    basename = f"traditional_ep{episode_idx:03d}"
    path = os.path.join(output_dir, f"{basename}.{video_format}")
    if video_format == "gif":
        imageio.mimsave(path, frames, fps=fps)
    else:
        imageio.mimwrite(path, frames, fps=fps, quality=8, codec="libx264")
    return path


def apply_obstacle_controller(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    sim_step: int,
    mode: str,
    alternate_steps: int,
    random_steps: int,
    state: ObstacleControllerState,
) -> None:
    if not OBSTACLE_ACTUATOR_IDS.size:
        return
    qpos_obs = data.qpos[list(OBSTACLE_JOINT_SLICE)]
    qvel_obs = data.qvel[list(OBSTACLE_JOINT_SLICE)]

    if mode == "none":
        data.ctrl[OBSTACLE_ACTUATOR_IDS] = 0
        return
    if mode == "reactive":
        try:
            q_slide = data.qpos[15]
            qd_slide = data.qvel[15]
            pos_agent = data.xpos[7]
            pos_obs = data.xpos[15]
            if abs(pos_agent[1] - pos_obs[1]) < 2:
                delta_pos_y = pos_agent[1] - pos_obs[1] - 3.0
                q_target_slide = q_slide - delta_pos_y
                tau_slide = Kp * (q_target_slide - q_slide) + Kd * (-qd_slide)
                data.ctrl[15] = tau_slide
            else:
                data.ctrl[15] = 0
        except Exception:
            data.ctrl[15] = 0
        return

    if mode == "random":
        if sim_step == 0 or (random_steps > 0 and sim_step % random_steps == 0):
            state.random_action = np.random.uniform(-1, 1, size=8).astype(np.float32)
        action = state.random_action
    elif mode == "static":
        action = DEFAULT_STATIC_ACTION
    elif mode == "alternate":
        idx = 0 if alternate_steps <= 0 else (sim_step // alternate_steps) % 2
        action = DEFAULT_ALTERNATE_ACTION1 if idx == 0 else DEFAULT_ALTERNATE_ACTION2
    else:
        action = np.zeros(8, dtype=np.float32)

    qpos_target = action_to_qpos_general(model, action, OBSTACLE_JOINT_SLICE)
    torque = Kp * (qpos_target - qpos_obs) + Kd * (-qvel_obs)
    ctrl_min = model.actuator_ctrlrange[OBSTACLE_ACTUATOR_IDS, 0]
    ctrl_max = model.actuator_ctrlrange[OBSTACLE_ACTUATOR_IDS, 1]
    data.ctrl[OBSTACLE_ACTUATOR_IDS] = np.clip(torque, ctrl_min, ctrl_max)


def advance_to_valid_target(path: List[Optional[np.ndarray]], start_idx: int) -> int:
    idx = start_idx
    while idx < len(path) and path[idx] is None:
        idx += 1
    return idx


def rollout_episode(
    episode_idx: int,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    data_plan: mujoco.MjData,
    renderer: Optional[mujoco.Renderer],
    viewer: Optional["mujoco.viewer.Handle"],
    args: argparse.Namespace,
    tip_id: int,
) -> Tuple[int, str, Optional[str]]:
    mujoco.mj_resetData(model, data)
    data.qvel[:] = 0
    data.ctrl[:] = 0
    mujoco.mj_forward(model, data)

    workspace_grid = np.zeros(GRID_SIZE, dtype=np.int8)
    fill_workspace_from_model(model, data, workspace_grid)

    data_plan.qpos[:] = data.qpos[:]
    data_plan.qvel[:] = 0
    mujoco.mj_forward(model, data_plan)

    start_ee = data.site_xpos[tip_id][:3].copy()
    goal_ee = sample_random_goal()
    print(f"[episode {episode_idx}] sampled goal: {goal_ee}")

    _, _, ee_qpos_path = plan_path_and_ik(model, data_plan, start_ee, goal_ee, workspace_grid)
    if not ee_qpos_path:
        print(f"[episode {episode_idx}] planner failed, skipping.")
        return 0, "plan_failed", None

    frames: List[np.ndarray] = []
    first_frame = capture_frame(renderer, data)
    if first_frame is not None:
        frames.append(first_frame)

    obstacle_state = ObstacleControllerState(
        random_action=np.random.uniform(-1, 1, size=8).astype(np.float32)
    )

    joint_ids = np.arange(CONTROLLABLE_JOINTS)
    actuator_ids = np.arange(CONTROLLABLE_JOINTS)
    path_idx = advance_to_valid_target(ee_qpos_path, 0)
    inner_step = 0
    sim_step = 0
    reason = "max_steps"

    while sim_step < args.max_steps:
        if viewer is not None and not viewer.is_running():
            return sim_step, "viewer_closed", None

        mujoco.mj_step(model, data)

        if path_idx >= len(ee_qpos_path):
            reason = "success"
            break

        qpos_target = ee_qpos_path[path_idx]
        if qpos_target is not None:
            qpos_now = data.qpos[joint_ids]
            qvel_now = data.qvel[joint_ids]
            torque = Kp * (qpos_target - qpos_now) + Kd * (0.0 - qvel_now)
            ctrl_min = model.actuator_ctrlrange[actuator_ids, 0]
            ctrl_max = model.actuator_ctrlrange[actuator_ids, 1]
            data.ctrl[actuator_ids] = np.clip(torque, ctrl_min, ctrl_max)

            inner_step += 1
            if inner_step >= STEP_PER_WAYPOINT:
                inner_step = 0
                path_idx = advance_to_valid_target(ee_qpos_path, path_idx + 1)
        else:
            path_idx = advance_to_valid_target(ee_qpos_path, path_idx + 1)

        apply_obstacle_controller(
            model=model,
            data=data,
            sim_step=sim_step,
            mode=args.obstacle_mode,
            alternate_steps=args.alternate_steps,
            random_steps=args.random_steps,
            state=obstacle_state,
        )

        frame = capture_frame(renderer, data)
        if frame is not None:
            frames.append(frame)

        if viewer is not None:
            viewer.sync()

        score = check_collision_point(model, data)
        if score:
            reason = "collision"
            break

        sim_step += 1

    saved_path = save_video(frames, args.output_dir, episode_idx, args.video_format, args.fps)
    summary = f"[episode {episode_idx}] steps={sim_step}, reason={reason}"
    if saved_path:
        summary += f", video={saved_path}"
    else:
        summary += ", video=unavailable"
    print(summary)
    return sim_step, reason, saved_path


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    data_plan = mujoco.MjData(model)

    tip_name = "tip8_agent"
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_name)
    if tip_id == -1:
        raise RuntimeError(f"Site {tip_name} not found in model.")

    renderer: Optional[mujoco.Renderer] = None
    try:
        renderer = mujoco.Renderer(model, width=args.width, height=args.height)
    except Exception as exc:
        print(f"[warn] Failed to create offscreen renderer, recording disabled: {exc}")

    viewer = None
    if args.show_viewer:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
        except Exception as exc:
            viewer = None
            print(f"[warn] Failed to open viewer window: {exc}")

    try:
        for ep in range(args.episodes):
            np.random.seed(args.seed + ep)
            steps, reason, _ = rollout_episode(
                episode_idx=ep,
                model=model,
                data=data,
                data_plan=data_plan,
                renderer=renderer,
                viewer=viewer,
                args=args,
                tip_id=tip_id,
            )
            if reason == "viewer_closed":
                print("[info] Viewer closed by user, stopping remaining episodes.")
                break
            if steps == 0 and reason == "plan_failed":
                continue
    finally:
        if viewer is not None:
            try:
                viewer.close()
            except Exception as exc:
                print(f"[warn] Failed to close viewer cleanly: {exc}")
        if renderer is not None:
            renderer.close()


if __name__ == "__main__":
    main()
