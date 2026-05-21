#!/usr/bin/env python
# coding: utf-8

# ik_solver.py
import numpy as np
import mujoco

def simple_ik(
    model,
    data_plan,
    target_pos,
    target_ori=None,      # ✅ 新增：外部指定目标姿态 (3x3 矩阵)
    ee_site_name="tip8_agent",
    max_iters=100,
    lr=0.1,
    tol=1e-4,
    controllable_joints=8,
    w_pos=1.0,
    w_ori=0.5,
    keep_ori=True
):
    """
    简单雅可比逆运动学求解器。

    参数：
        model: mujoco.MjModel
        data_plan: mujoco.MjData
        target_pos: np.ndarray(3,)
        target_ori: np.ndarray(3,3) 或 None, 目标末端姿态 (若为None且keep_ori=True，则使用初始姿态)
        ee_site_name: str, 末端site名称
        max_iters: int
        lr: float
        tol: float
        controllable_joints: int
        w_pos, w_ori: float, 权重
        keep_ori: bool, 是否在没有目标姿态时保持初始姿态
    返回：
        qpos[:controllable_joints]
    """
    qpos = data_plan.qpos.copy()
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site_name)
    if ee_site_id == -1:
        raise ValueError(f"Site {ee_site_name} not found in model")

    # ---------- 若目标姿态未提供，则根据keep_ori确定 ----------
    if target_ori is None and keep_ori:
        mujoco.mj_forward(model, data_plan)
        target_ori = data_plan.site_xmat[ee_site_id].reshape(3, 3)
    elif target_ori is None:
        keep_ori = False

    for _ in range(max_iters):
        data_plan.qpos[:] = qpos
        mujoco.mj_forward(model, data_plan)

        # 当前末端位姿
        ee_pos = data_plan.site_xpos[ee_site_id]
        ee_mat = data_plan.site_xmat[ee_site_id].reshape(3, 3)

        # 位置误差
        pos_err = target_pos - ee_pos

        # 姿态误差
        if keep_ori:
            R_err = 0.5 * (np.cross(ee_mat[:, 0], target_ori[:, 0]) +
                           np.cross(ee_mat[:, 1], target_ori[:, 1]) +
                           np.cross(ee_mat[:, 2], target_ori[:, 2]))
        else:
            R_err = np.zeros(3)

        if np.linalg.norm(pos_err) < tol and np.linalg.norm(R_err) < tol:
            break

        # 雅可比
        Jp_full = np.zeros((3, model.nv))
        Jr_full = np.zeros((3, model.nv))
        mujoco.mj_jacSite(model, data_plan, Jp_full, Jr_full, ee_site_id)
        Jp = Jp_full[:, :controllable_joints]
        Jr = Jr_full[:, :controllable_joints]

        # 拼接误差
        error = np.hstack((w_pos * pos_err, w_ori * R_err))
        J_full = np.vstack((w_pos * Jp, w_ori * Jr))

        dq = lr * np.linalg.pinv(J_full) @ error
        qpos[:controllable_joints] += dq

        # 限制关节范围
        for j in range(controllable_joints):
            qpos[j] = np.clip(qpos[j], model.jnt_range[j, 0], model.jnt_range[j, 1])

    return qpos[:controllable_joints]
