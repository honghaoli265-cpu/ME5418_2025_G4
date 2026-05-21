#!/usr/bin/env python
# coding: utf-8
"""
main_get_expert_dataset3.py（带视频保存版，适配 MuJoCo 3.3.6）
"""

import os
import csv
import numpy as np
import mujoco
import mujoco.viewer
from dstar_lite import DStarLite
from ik_solver import simple_ik
import imageio

# ----------------------------
# 配置（可修改）
# ----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(BASE_DIR)
XML_PATH = os.path.join(PARENT_DIR, "Fencing_agent&obstacle_description", "fencing_arm_ver3.xml")

episode_num = 1
USE_RENDER = True  # True: 启动窗口渲染，False: 不渲染

GRID_SIZE = (100, 20, 20)
EE_MIN = np.array([-4, -0.5, -0.5])
EE_MAX = np.array([1, 0.5, 0.5])

OBS_BODY_LIST = [f"link{i}_obstacle" for i in range(8)]
OBS_UPDATE_INTERVAL = 200

Kp = 50.0
Kd = 1.5
CONTROLLABLE_JOINTS = 8
STEP_PER_WAYPOINT = 200

OBSTACLE_MODE = 'alternate'
ALTERNATE_STEPS = 1000
ALTERNATE_ACTION1 = np.array([0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ALTERNATE_ACTION2 = np.array([0.0, 0.9, 0.6, -0.7, -0.7, 0.5, 0.3, 0.3], dtype=np.float32)
RANDOM_STEPS = 1000
RANDOM_ACTION = np.random.uniform(-1, 1, size=8).astype(np.float32)

# ------------------------
# 视频保存配置
VIDEO_SAVE = True
VIDEO_FORMAT = "mp4"  # "mp4" 或 "gif"
VIDEO_FPS = 30
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_OUTPUT_DIR = os.path.join(BASE_DIR, "rollout_videos")
os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)


def create_renderer(mj_model):
    return mujoco.Renderer(mj_model, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)


def capture_frame(offscreen_renderer, mj_data):
    offscreen_renderer.update_scene(mj_data)
    frame = offscreen_renderer.render()
    return np.clip(frame * 255, 0, 255).astype(np.uint8)

# ------------------------
# 工具函数
def sample_random_goal():
    x = np.random.uniform(-3.5, -2.5)
    y = np.random.uniform(-0.1, 0.1)
    z = np.random.uniform(-0.2, 0.2)
    return np.array([x, y, z])

def ee2grid(ee_pos):
    grid_pos = (ee_pos - EE_MIN) / (EE_MAX - EE_MIN) * (np.array(GRID_SIZE) - 1)
    return tuple(np.clip(grid_pos.astype(int), 0, np.array(GRID_SIZE)-1))

def grid2ee(grid_pos):
    ee_pos = np.array(grid_pos) / (np.array(GRID_SIZE) - 1) * (EE_MAX - EE_MIN) + EE_MIN
    return ee_pos

def action_to_qpos_general(model, action_in, joint_slice):
    if isinstance(joint_slice, slice):
        j_ids = np.arange(joint_slice.start, joint_slice.stop)
    else:
        j_ids = np.array(joint_slice)
    joint_range = model.jnt_range[j_ids]
    q_min = joint_range[:, 0]
    q_max = joint_range[:, 1]
    q_target = q_min + (action_in + 1) * 0.5 * (q_max - q_min)
    return q_target

def fill_workspace_from_model(model, data, workspace_grid):
    workspace_grid.fill(0)
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
            grid_min = np.clip(((lower - EE_MIN) / (EE_MAX - EE_MIN) * (np.array(GRID_SIZE) - 1)).astype(int), 0, np.array(GRID_SIZE)-1)
            grid_max = np.clip(((upper - EE_MIN) / (EE_MAX - EE_MIN) * (np.array(GRID_SIZE) - 1)).astype(int), 0, np.array(GRID_SIZE)-1)
            workspace_grid[grid_min[0]:grid_max[0]+1, grid_min[1]:grid_max[1]+1, grid_min[2]:grid_max[2]+1] = 1
    return workspace_grid

def plan_path_and_ik(model, data_plan, start_ee, goal_ee, workspace_grid):
    start_node = ee2grid(start_ee)
    goal_node = ee2grid(goal_ee)
    dstar = DStarLite(start_node, goal_node, workspace_grid.copy())
    dstar.compute_shortest_path()
    path_nodes = dstar.get_path()
    if not path_nodes:
        print("⚠️ D* Lite 未找到路径")
        return None, None, None

    ee_path = np.array([grid2ee(node) for node in path_nodes])
    mujoco.mj_forward(model, data_plan)
    tip_name = "tip8_agent"
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_name)
    target_ori = data_plan.site_xmat[tip_id].reshape(3, 3)

    ee_qpos_path = []
    for p in ee_path:
        q = simple_ik(model, data_plan, np.array(p), target_ori=target_ori, controllable_joints=CONTROLLABLE_JOINTS)
        ee_qpos_path.append(q)
    return path_nodes, ee_path, ee_qpos_path

def check_collision_point(model, data):
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
        elif cond1 or cond2:
            return 2
        elif cond3 or cond4:
            return 1
    return 0

# =======================
# 主程序
# =======================
def main():
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    data_plan = mujoco.MjData(model)
    workspace_grid = np.zeros(GRID_SIZE, dtype=np.int8)
    workspace_grid = fill_workspace_from_model(model, data, workspace_grid)

    tip_name = "tip8_agent"
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_name)
    assert tip_id != -1, f"Site {tip_name} not found"

    # obstacle 模式
    obstacle_mode = OBSTACLE_MODE
    alternate_steps = ALTERNATE_STEPS
    alternate_action1 = ALTERNATE_ACTION1.copy()
    alternate_action2 = ALTERNATE_ACTION2.copy()
    random_steps = RANDOM_STEPS
    random_action = RANDOM_ACTION.copy()

    # CSV 初始化
    csv_path = os.path.join(BASE_DIR, "expert_dataset_from_dstar_lite.csv")
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    header = (
        ["episode", "step"] +
        [f"qpos{i}" for i in range(8)] +
        [f"qvel{i}" for i in range(8)] +
        [f"qpos{i}" for i in range(12, 20)] +
        [f"action{i}" for i in range(8)] +
        ["terminated"]
    )
    csv_writer.writerow(header)

    # GUI viewer
    viewer = mujoco.viewer.launch_passive(model, data) if USE_RENDER else None

    # 离屏渲染 Renderer
    renderer = create_renderer(model) if VIDEO_SAVE else None

    try:
        for episode_idx in range(episode_num):
            print(f"\n========== 🚀 Episode {episode_idx + 1} ==========")
            mujoco.mj_resetData(model, data)
            data.qvel[:] = 0
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)

            workspace_grid = fill_workspace_from_model(model, data, workspace_grid)
            data_plan.qpos[:] = data.qpos[:]
            data_plan.qvel[:] = 0
            mujoco.mj_forward(model, data_plan)

            start_ee = data.site_xpos[tip_id][:3].copy()
            goal_ee = sample_random_goal()
            print(f"🎯 目标点: {goal_ee}")

            path_nodes, ee_path, ee_qpos_path = plan_path_and_ik(model, data_plan, start_ee, goal_ee, workspace_grid)
            if path_nodes is None:
                print("⚠️ 初始规划失败，跳过 episode")
                continue

            joint_ids = np.arange(CONTROLLABLE_JOINTS)
            actuator_ids = np.arange(CONTROLLABLE_JOINTS)
            obs_joint_slice = range(12, 20)
            obs_actuator_ids = np.arange(8, 16)
            path_idx = 0
            inner_step = 0
            sim_step = 0

            frames = []

            while viewer is None or viewer.is_running():
                mujoco.mj_step(model, data)

                # qpos_target / action
                if ee_qpos_path and path_idx < len(ee_qpos_path) and ee_qpos_path[path_idx] is not None:
                    qpos_target = ee_qpos_path[path_idx]
                else:
                    qpos_target = np.full(8, np.nan)

                if not np.any(np.isnan(qpos_target)):
                    jnt_range = model.jnt_range[:CONTROLLABLE_JOINTS]
                    qpos_min = jnt_range[:, 0]
                    qpos_max = jnt_range[:, 1]
                    action = 2 * (qpos_target - qpos_min) / (qpos_max - qpos_min) - 1
                else:
                    action = np.full(CONTROLLABLE_JOINTS, np.nan)

                # 碰撞检测
                score = check_collision_point(model, data)
                terminated = 1 if score else 0

                # 写入 CSV
                state_row = np.concatenate([
                    [episode_idx, sim_step],
                    data.qpos[:8],
                    data.qvel[:8],
                    data.qpos[12:20],
                    action,
                    [terminated]
                ])
                csv_writer.writerow(state_row.tolist())

                if terminated == 1 or sim_step > 20000 or (ee_qpos_path and path_idx >= len(ee_qpos_path)):
                    break

                # Agent PD 控制
                if ee_qpos_path and path_idx < len(ee_qpos_path) and ee_qpos_path[path_idx] is not None:
                    qpos_now = data.qpos[joint_ids]
                    qvel_now = data.qvel[joint_ids]
                    qvel_target = np.zeros_like(qvel_now)
                    torque = Kp * (qpos_target - qpos_now) + Kd * (qvel_target - qvel_now)
                    ctrl_min = model.actuator_ctrlrange[actuator_ids, 0]
                    ctrl_max = model.actuator_ctrlrange[actuator_ids, 1]
                    data.ctrl[actuator_ids] = np.clip(torque, ctrl_min, ctrl_max)
                    inner_step += 1
                    if inner_step >= STEP_PER_WAYPOINT:
                        inner_step = 0
                        path_idx += 1
                        if path_idx >= len(ee_qpos_path):
                            break

                # Obstacle 控制
                qpos_obs_now = data.qpos[list(obs_joint_slice)]
                qvel_obs_now = data.qvel[list(obs_joint_slice)]
                if obstacle_mode == 'random':
                    if sim_step % random_steps == 0:
                        random_action = np.random.uniform(-1, 1, size=8).astype(np.float32)
                    action_obstacle = np.clip(random_action, -1, 1)
                elif obstacle_mode == 'none':
                    action_obstacle = np.zeros(8)
                elif obstacle_mode == 'static':
                    action_obstacle = np.array([0.0, -0.9, 0.6, -0.5, -0.5, 0.5, 0.0, 0.0], dtype=np.float32)
                elif obstacle_mode == 'alternate':
                    idx = (sim_step // alternate_steps) % 2
                    action_obstacle = alternate_action1 if idx == 0 else alternate_action2
                else:
                    action_obstacle = np.zeros(8)

                qpos_target_obs = action_to_qpos_general(model, action_obstacle, obs_joint_slice)
                qvel_target_obs = np.zeros(8)
                torque_obs = Kp * (qpos_target_obs - qpos_obs_now) + Kd * (qvel_target_obs - qvel_obs_now)
                ctrl_min_obs = model.actuator_ctrlrange[obs_actuator_ids, 0]
                ctrl_max_obs = model.actuator_ctrlrange[obs_actuator_ids, 1]
                data.ctrl[obs_actuator_ids] = np.clip(torque_obs, ctrl_min_obs, ctrl_max_obs)

                # 渲染 & 捕获帧
                if USE_RENDER and viewer is not None:
                    viewer.sync()

                if VIDEO_SAVE and renderer is not None:
                    try:
                        frame_uint8 = capture_frame(renderer, data)
                        frames.append(frame_uint8)
                    except Exception as e:
                        print(f"⚠️ capture frame exception at sim_step {sim_step}: {e}")

                sim_step += 1

            # 保存视频
            if VIDEO_SAVE and len(frames) > 0:
                basename = f"rollout_ep{episode_idx:03d}"
                path = os.path.join(VIDEO_OUTPUT_DIR, f"{basename}.{VIDEO_FORMAT}")
                if VIDEO_FORMAT == "gif":
                    imageio.mimsave(path, frames, fps=VIDEO_FPS)
                else:
                    with imageio.get_writer(
                        path,
                        fps=VIDEO_FPS,
                        codec="libx264",
                        format="mp4",
                        pixelformat="yuv420p",
                        macro_block_size=None,
                    ) as writer:
                        for frame in frames:
                            writer.append_data(frame)
                print(f"[video] Episode {episode_idx} 视频已保存到 {path}")
            elif VIDEO_SAVE:
                print(f"⚠️ Episode {episode_idx} 没有捕获到任何帧，视频未保存")

            print(f"Episode {episode_idx + 1} finished (sim steps: {sim_step}).")

    finally:
        csv_file.close()
        if viewer is not None:
            try:
                viewer.close()
            except Exception as e:
                print("关闭 viewer 出错：", e)
        if renderer is not None:
            try:
                renderer.close()
            except:
                pass
        print("🛑 程序结束，CSV 已保存到:", csv_path)


if __name__ == "__main__":
    main()
