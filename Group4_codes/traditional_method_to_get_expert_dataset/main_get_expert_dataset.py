#!/usr/bin/env python
# coding: utf-8
"""
main_get_expert_dataset_merged.py

合并说明：
- 基于你提供的 main_get_expert_dataset.py（A），增加 obstacle 的
  alternate / random 模式（控制关节 12:20，actuator 8:16）。
- 其他逻辑尽量保持原样，仅做必要修正以保证可运行。
"""

import os
import time
import csv
import numpy as np
import mujoco
import mujoco.viewer
from dstar_lite import DStarLite
from ik_solver import simple_ik

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

# 动态刷新参数（读取 model 中的 body）
OBS_BODY_LIST = [f"link{i}_obstacle" for i in range(8)]
OBS_UPDATE_INTERVAL = 200  # 每多少仿真步检查一次障碍（帧）
REPLAN_ON_CHANGE = False  # 障碍变化时是否立即重规划
REPLAN_ON_PATH_BLOCK = True  # 路径被障碍阻断时重规划

# 控制参数
Kp = 50.0
Kd = 1.5
CONTROLLABLE_JOINTS = 8
STEP_PER_WAYPOINT = 200  # 每个 IK 目标点执行多少步

# ----------------------------
# Obstacle movement 模式配置（新增）
# ----------------------------
# 可选模式： 'static', 'none', 'random', 'reactive', 'alternate'
OBSTACLE_MODE = 'random'  # 默认：alternate（你可改为 'random' / 'static' / 'none' / 'reactive'）

# alternate 模式参数
ALTERNATE_STEPS = 1000
ALTERNATE_ACTION1 = np.array([0.0, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
ALTERNATE_ACTION2 = np.array([0.0, 0.9, 0.6, -0.7, -0.7, 0.5, 0.3, 0.3], dtype=np.float32)

# random 模式参数
RANDOM_STEPS = 1000
# 初始随机动作
RANDOM_ACTION = np.random.uniform(-1, 1, size=8).astype(np.float32)

# ------------------------
# 生成随机目标点（指定范围）
# ------------------------
def sample_random_goal():
    x = np.random.uniform(-3.5, -2.5)
    y = np.random.uniform(-0.1, 0.1)
    z = np.random.uniform(-0.2, 0.2)
    return np.array([x, y, z])


# ----------------------------
# 工具函数：坐标映射
# ----------------------------
def ee2grid(ee_pos):
    """将 EE 实际坐标映射到栅格坐标（整数三元组）"""
    grid_pos = (ee_pos - EE_MIN) / (EE_MAX - EE_MIN) * (np.array(GRID_SIZE) - 1)
    return tuple(np.clip(grid_pos.astype(int), 0, np.array(GRID_SIZE)-1))

def grid2ee(grid_pos):
    """栅格坐标映射回 EE 实际坐标"""
    ee_pos = np.array(grid_pos) / (np.array(GRID_SIZE) - 1) * (EE_MAX - EE_MIN) + EE_MIN
    return ee_pos

# ----------------------------
# 通用 action -> qpos 映射（支持不同关节切片）
# ----------------------------
def action_to_qpos_general(model, action_in, joint_slice):
    """
    action_in: array in [-1,1] of length == len(joint_slice)
    joint_slice: slice or list/array of joint indices to map (e.g. range(12,20))
    """
    if isinstance(joint_slice, slice):
        j_ids = np.arange(joint_slice.start, joint_slice.stop)
    else:
        j_ids = np.array(joint_slice)
    joint_range = model.jnt_range[j_ids]
    q_min = joint_range[:, 0]
    q_max = joint_range[:, 1]
    q_target = q_min + (action_in + 1) * 0.5 * (q_max - q_min)
    return q_target

# ----------------------------
# 填充 workspace_grid（把 model 中障碍写入栅格）
# ----------------------------
def fill_workspace_from_model(model, data, workspace_grid):
    """扫描 model 中的 body->geom，将其写入 workspace_grid（以包围盒近似）"""
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

# ----------------------------
# 检测路径是否被障碍占用
# ----------------------------
def path_blocked_by_grid(path_grid_nodes, workspace_grid):
    """path_grid_nodes: list/iterable of grid tuple positions"""
    for node in path_grid_nodes:
        x, y, z = node
        if workspace_grid[x, y, z] == 1:
            return True
    return False

# ----------------------------
# 生成 D* Lite 路径及 IK 关节解
# ----------------------------
def plan_path_and_ik(model, data_plan, start_ee, goal_ee, workspace_grid):
    start_node = ee2grid(start_ee)
    goal_node = ee2grid(goal_ee)
    dstar = DStarLite(start_node, goal_node, workspace_grid.copy())
    dstar.compute_shortest_path()
    path_nodes = dstar.get_path()
    if not path_nodes:
        print("⚠️ D* Lite 未找到路径（path empty）")
        return None, None, None

    ee_path = np.array([grid2ee(node) for node in path_nodes])

    # ---------- 获取目标末端姿态（XML默认状态） ----------
    mujoco.mj_forward(model, data_plan)  # data_plan 初始化为 XML 默认 qpos
    tip_name = "tip8_agent"
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_name)
    target_ori = data_plan.site_xmat[tip_id].reshape(3, 3)  # XML 默认末端姿态

    # IK：为每个 ee 位置求可控关节解
    ee_qpos_path = []
    for p in ee_path:
        q = simple_ik(
            model,
            data_plan,
            np.array(p),
            target_ori=target_ori,           # ✅ 传入 XML 默认姿态
            controllable_joints=CONTROLLABLE_JOINTS
        )
        if q is None:
            print("⚠️ IK 未能为某点求解，放弃该路径点")
            ee_qpos_path.append(None)
        else:
            ee_qpos_path.append(q)

    return path_nodes, ee_path, ee_qpos_path

# ----------------------------
# 碰撞检测 / 得分判断（逐点）
# ----------------------------
def check_collision_point(model, data):
    """
    检查当前 step 的所有接触信息并返回得分：
    0: 无碰撞
    1: obstacle 得分
    2: agent 得分
    3: 双方同时得分
    """
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

# ----------------------------
# 无得分碰撞检测（仅判断碰撞，不计分）
# ----------------------------
def check_collision_no_point(model, data):
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

        cond1 = (contype1 == 4 and "agent" in name1 and contype2 == 18 and "obstacle" in name2)
        cond2 = (contype2 == 4 and "agent" in name2 and contype1 == 18 and "obstacle" in name1)
        cond3 = (contype1 == 8 and "obstacle" in name1 and contype2 == 18 and "agent" in name2)
        cond4 = (contype2 == 8 and "obstacle" in name2 and contype1 == 18 and "agent" in name1)
        cond5 = (contype1 == 18 and "agent" in name1 and contype2 == 18 and "obstacle" in name2)
        cond6 = (contype2 == 18 and "agent" in name2 and contype1 == 18 and "obstacle" in name1)

        if (cond1 or cond2) or (cond3 or cond4) or (cond5 or cond6):
            return True
    return False
# =======================
# 主程序 main()
# =======================
def main():
    # =========================
    # 初始化模型与仿真数据
    # =========================
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)            # ✅ 只创建一次 data
    data_plan = mujoco.MjData(model)       # 用于规划

    # 工作栅格
    workspace_grid = np.zeros(GRID_SIZE, dtype=np.int8)
    workspace_grid = fill_workspace_from_model(model, data, workspace_grid)

    tip_name = "tip8_agent"
    tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, tip_name)
    assert tip_id != -1, f"Site {tip_name} not found"

    agent_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tip8_agent")
    obstacle_base_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link_obstacle")
    assert agent_body_id != -1, "Body tip8_agent not found"
    assert obstacle_base_body_id != -1, "Body base_link_obstacle not found"

    # 初始化 obstacle 模式相关变量（从全局参数）
    obstacle_mode = OBSTACLE_MODE
    alternate_steps = ALTERNATE_STEPS
    alternate_action1 = ALTERNATE_ACTION1.copy()
    alternate_action2 = ALTERNATE_ACTION2.copy()
    random_steps = RANDOM_STEPS
    random_action = RANDOM_ACTION.copy()

    # =========================
    # 初始化 CSV 文件
    # =========================
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

    # =========================
    # 启动 viewer（可选）
    # =========================
    viewer = None
    if USE_RENDER:
        print("✅ 启动 MuJoCo viewer，开始执行")
        viewer = mujoco.viewer.launch_passive(model, data)
    else:
        print("⚠️ 渲染已关闭，运行无窗口模式")

    try:
        for episode_idx in range(episode_num):
            print(f"\n========== 🚀 开始第 {episode_idx + 1} 个 episode ==========")

            # Reset 主仿真
            mujoco.mj_resetData(model, data)
            data.qvel[:] = 0
            data.ctrl[:] = 0
            mujoco.mj_forward(model, data)

            # 填充 workspace_grid
            workspace_grid = np.zeros(GRID_SIZE, dtype=np.int8)
            workspace_grid = fill_workspace_from_model(model, data, workspace_grid)

            # 规划用 data_plan 初始化
            data_plan.qpos[:] = data.qpos[:]
            data_plan.qvel[:] = 0
            mujoco.mj_forward(model, data_plan)

            workspace_grid = fill_workspace_from_model(model, data, workspace_grid)
            start_ee = data.site_xpos[tip_id][:3].copy()
            goal_ee = sample_random_goal()
            print(f"🎯 Episode {episode_idx + 1} 目标点: {goal_ee}")

            path_nodes, ee_path, ee_qpos_path = plan_path_and_ik(
                model, data_plan, start_ee, goal_ee, workspace_grid
            )
            if path_nodes is None:
                print("⚠️ 初始规划失败，跳过该 episode")
                continue

            joint_ids = np.arange(CONTROLLABLE_JOINTS)
            actuator_ids = np.arange(CONTROLLABLE_JOINTS)
            prev_grid_snapshot = workspace_grid.copy()
            path_idx = 0
            inner_step = 0
            sim_step = 0
            episode_step = 0
            episode_reward = 0.0

            # Episode 主循环
            while (viewer is None or viewer.is_running()):
                # -------------------------
                # 先推进仿真一步（原代码在循环顶部做 mj_step）
                # -------------------------
                mujoco.mj_step(model, data)

                # ---------- 获取 qpos_target ----------
                if ee_qpos_path and path_idx < len(ee_qpos_path) and ee_qpos_path[path_idx] is not None:
                    qpos_target = ee_qpos_path[path_idx]
                else:
                    qpos_target = np.full(8, np.nan)

                # ---------- 归一化为 action ----------
                if not np.any(np.isnan(qpos_target)):
                    jnt_range = model.jnt_range[:CONTROLLABLE_JOINTS]
                    qpos_min = jnt_range[:, 0]
                    qpos_max = jnt_range[:, 1]
                    action = 2 * (qpos_target - qpos_min) / (qpos_max - qpos_min) - 1
                else:
                    action = np.full(CONTROLLABLE_JOINTS, np.nan)

                # ---------- 碰撞检测 ----------
                score = check_collision_point(model, data)
                terminated = 1 if score else 0

                # ---------- 奖励计算 ----------
                agent_ee_pos = data.xpos[agent_body_id][:3]
                target_pos = data.xpos[obstacle_base_body_id][:3]
                dist_to_target = np.linalg.norm(agent_ee_pos - target_pos)
                R_dist = dist_to_target ** 2
                w_dist = -0.1

                ccp = score
                R_success = 1.0 if ccp in (2, 3) else 0.0
                w_success = 10000.0
                R_failure = 1.0 if ccp in (1, 3) else 0.0
                w_failure = -10000.0

                ccnp = check_collision_no_point(model, data)
                R_collision = 1.0 if ccnp else 0.0
                w_collision = 0.0

                action_for_reward = np.nan_to_num(action, nan=0.0)
                R_action = np.sum(np.square(action_for_reward))
                w_action = -0.01

                R_step = 1.0
                w_step = -0.1

                reward = (
                    w_dist * R_dist
                    + w_success * R_success
                    + w_failure * R_failure
                    + w_collision * R_collision
                    + w_action * R_action
                    + w_step * R_step
                )
                episode_reward += reward

                # ---------- 写入CSV ----------
                state_row = np.concatenate([
                    [episode_idx, sim_step],
                    data.qpos[:8],
                    data.qvel[:8],
                    data.qpos[12:20],
                    action,
                    [terminated]
                ])
                csv_writer.writerow(state_row.tolist())

                # ---------- 检查终止条件 ----------
                if terminated == 1:
                    print(f"🚨 第 {episode_idx + 1} 个 episode 在 step {sim_step} 因terminated终止。")
                    break
                if sim_step > 20_000:
                    print(f"🚨 第 {episode_idx + 1} 个 episode 在 step {sim_step} 因truncated终止。")
                    break

                # ---------- Agent 控制执行（PD） ----------
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
                            print("✅ 到达路径末端，结束该 episode。")
                            break
                else:
                    # 没有有效 qpos_target 时不控制 agent（保持现状或让物理回落）
                    pass

                # ---------- Obstacle 模式控制（新增） ----------
                # obstacle joints: indices 12..19 (inclusive)  -> joint slice range(12,20)
                # obstacle actuators: indices 8..15
                obs_joint_slice = range(12, 20)
                obs_actuator_ids = np.arange(8, 16)
                qpos_obs_now = data.qpos[list(obs_joint_slice)]
                qvel_obs_now = data.qvel[list(obs_joint_slice)]

                if obstacle_mode == 'random':
                    # 每隔 random_steps 重新随机动作（基于 sim_step）
                    if sim_step % random_steps == 0:
                        random_action = np.random.uniform(-1, 1, size=8).astype(np.float32)
                    action_obstacle = np.clip(random_action, -1, 1)
                    qpos_target_obs = action_to_qpos_general(model, action_obstacle, obs_joint_slice)
                    qvel_target_obs = np.zeros(8)
                    torque_obs = Kp * (qpos_target_obs - qpos_obs_now) + Kd * (qvel_target_obs - qvel_obs_now)
                    ctrl_min_obs = model.actuator_ctrlrange[obs_actuator_ids, 0]
                    ctrl_max_obs = model.actuator_ctrlrange[obs_actuator_ids, 1]
                    data.ctrl[obs_actuator_ids] = np.clip(torque_obs, ctrl_min_obs, ctrl_max_obs)

                elif obstacle_mode == 'none':
                    # 只有重力，输出扭矩为 0（不施加额外控制）
                    data.ctrl[obs_actuator_ids] = 0

                elif obstacle_mode == 'reactive':
                    # 简单躲避示例：监测 agent tip 与 obstacle base y 距离，触发第 15 个关节位移（示例）
                    # 保持以前的 reactive 实现风格
                    try:
                        q_slide = data.qpos[15]
                        qd_slide = data.qvel[15]
                        pos7 = data.xpos[7]    # agent 某 body
                        pos15 = data.xpos[15]  # obstacle 某 body
                        if abs(pos7[1] - pos15[1]) < 2:
                            delta_pos_y = pos7[1] - pos15[1] - 3
                            q_target_slide = q_slide - delta_pos_y
                            qd_target = 0.0
                            tau_slide = Kp * (q_target_slide - q_slide) + Kd * (qd_target - qd_slide)
                            data.ctrl[15] = tau_slide
                        else:
                            data.ctrl[15] = 0
                    except Exception:
                        # 若 body 索引/名称与你的 XML 不完全匹配，忽略 reactive 控制
                        data.ctrl[15] = 0

                elif obstacle_mode == 'static':
                    action_obstacle = np.array([0.0, -0.9, 0.6, -0.5, -0.5, 0.5, 0.0, 0.0], dtype=np.float32)
                    action_obstacle = np.clip(action_obstacle, -1, 1)
                    qpos_target_obs = action_to_qpos_general(model, action_obstacle, obs_joint_slice)
                    qvel_target_obs = np.zeros(8)
                    torque_obs = Kp * (qpos_target_obs - qpos_obs_now) + Kd * (qvel_target_obs - qvel_obs_now)
                    ctrl_min_obs = model.actuator_ctrlrange[obs_actuator_ids, 0]
                    ctrl_max_obs = model.actuator_ctrlrange[obs_actuator_ids, 1]
                    data.ctrl[obs_actuator_ids] = np.clip(torque_obs, ctrl_min_obs, ctrl_max_obs)

                elif obstacle_mode == 'alternate':
                    idx = (sim_step // alternate_steps) % 2
                    if idx == 0:
                        action_obstacle = alternate_action1
                    else:
                        action_obstacle = alternate_action2
                    action_obstacle = np.clip(action_obstacle, -1, 1)
                    qpos_target_obs = action_to_qpos_general(model, action_obstacle, obs_joint_slice)
                    qvel_target_obs = np.zeros(8)
                    torque_obs = Kp * (qpos_target_obs - qpos_obs_now) + Kd * (qvel_target_obs - qvel_obs_now)
                    ctrl_min_obs = model.actuator_ctrlrange[obs_actuator_ids, 0]
                    ctrl_max_obs = model.actuator_ctrlrange[obs_actuator_ids, 1]
                    data.ctrl[obs_actuator_ids] = np.clip(torque_obs, ctrl_min_obs, ctrl_max_obs)

                else:
                    # fallback: no control
                    data.ctrl[obs_actuator_ids] = 0

                # ---------- Render sync ----------
                if USE_RENDER and viewer is not None:
                    viewer.sync()

                # ---------- 步数更新 ----------
                sim_step += 1
                episode_step += 1

            print(f"Episode {episode_idx + 1} finished (sim steps: {sim_step}).")
            print(f"Episode {episode_idx + 1} total reward: {episode_reward:.3f}")

        print(f"\n🎯 所有 {episode_num} 个 episode 已完成。")

    finally:
        csv_file.close()
        if viewer is not None:
            try:
                viewer.close()
            except Exception as e:
                print("关闭 viewer 时出错：", e)
        print("🛑 程序结束，viewer 已关闭。CSV 已保存到:", csv_path)


if __name__ == "__main__":
    main()
