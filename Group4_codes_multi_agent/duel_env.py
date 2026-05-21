# duel_env.py

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# =======================
# Env / 环境
# =======================
class DuelEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    # =======================
    # Initialization / 初始化
    # =======================
    def __init__(self, xml_path, render_mode="human"):
        super().__init__()
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        self.time = 0.0
        self.render_mode = render_mode
        self.viewer = None  # passive viewer (GUI)
        self.cam = None     # for rgb_array mode
        self.max_episode_steps =100_000  # enforce 10k-step horizon
        self.episode_step = 0
       
        # =======================
        # Action space (8+8dim): 
        # 动作空间(8+8dim): agent 的 8 个 qpos + obstacle(agent2) 的 8 个 qpos
        # =======================
        action_dim = 8 + 8
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        
        # =======================
        # Observation space (24d+24im): 
        # agent robotic arm with 8 joints qpos, 8 joints qvel, obstacle(agent2) with 8 joints qpos
        # obstacle(agent2) robotic arm with 8 joints qpos, 8 joints qvel, agent with 8 joints qpos
        # 观测空间（24+24dim）：
        # =======================
        obs_dim = 24 + 24
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # =======================
        # 
        # 每步时间
        # =======================
        self.dt = 0.002
        
        # =======================
        # PD control parameters
        # PD 控制参数
        # ======================= 
        self.Kp1 = 30.0
        self.Kd1 = 10.0
        self.Kp2 = 30.0
        self.Kd2 = 10.0 
        
    # =======================
    #  / 归一化映射
    # =======================
    def action_to_qpos(self, action,joint_ids):
        joint_range = self.model.jnt_range[joint_ids]
        q_min = joint_range[:, 0]
        q_max = joint_range[:, 1]
        q_target = q_min + (action + 1) * 0.5 * (q_max - q_min)
        return q_target
    
    
    # =======================
    # Check collision for points / 得分检测
    # =======================
    
    def _check_collision_point(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get geom ID
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            # Get contype
            contype1 = self.model.geom_contype[geom1_id]
            contype2 = self.model.geom_contype[geom2_id]

            # Get body name
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            # name1 = self.model.body_id2name(body1_id)
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            # name2 = self.model.body_id2name(body2_id)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            # Agent ee ↔ Obstacle torso
            cond1 = (contype1 == 4 and "agent" in name1 and contype2 == 17 and "obstacle" in name2)
            cond2 = (contype2 == 4 and "agent" in name2 and contype1 == 17 and "obstacle" in name1)
            # Obstacle ee ↔ Agent torso
            cond3 = (contype1 == 8 and "obstacle" in name1 and contype2 == 17 and "agent" in name2)
            cond4 = (contype2 == 8 and "obstacle" in name2 and contype1 == 17 and "agent" in name1)
            
            if ( cond1 or cond2 ) and ( cond3 or cond4 ): # both bull point / 双方同时得分
                return 3
            elif  cond1 or cond2 : # agent bull point / agent 得分
                return 2
            elif  cond3 or cond4 : # obstacle bull point / obstacle 得分
                return 1
            else :
                pass
        return 0
    
    # =======================
    # Check collision for no points /不得分情况的碰撞检测
    # =======================
    
    def _check_collision_no_point(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            # Get geom ID
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            # Get contype
            contype1 = self.model.geom_contype[geom1_id]
            contype2 = self.model.geom_contype[geom2_id]

            # Get body name
            body1_id = self.model.geom_bodyid[geom1_id]
            body2_id = self.model.geom_bodyid[geom2_id]
            # name1 = self.model.body_id2name(body1_id)
            name1 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body1_id)
            # name2 = self.model.body_id2name(body2_id)
            name2 = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body2_id)

            # Agent ee ↔ Obstacle arm
            cond1 = (contype1 == 4 and "agent" in name1 and contype2 == 18 and "obstacle" in name2)
            cond2 = (contype2 == 4 and "agent" in name2 and contype1 == 18 and "obstacle" in name1)
            # Obstacle ee ↔ Agent arm
            cond3 = (contype1 == 8 and "obstacle" in name1 and contype2 == 18 and "agent" in name2)
            cond4 = (contype2 == 8 and "obstacle" in name2 and contype1 == 18 and "agent" in name1)
            # Agent arm ↔ Obstacle arm
            cond5 = (contype1 == 18 and "agent" in name1 and contype2 == 18 and "obstacle" in name2)
            cond6 = (contype2 == 18 and "agent" in name2 and contype1 == 18 and "obstacle" in name1)

            if ( cond1 or cond2 ) or ( cond3 or cond4 ) or ( cond5 or cond6 ): # collision for no points / 不得分的碰撞
                return True
            else :
                pass
            
        return False
    
    # =======================
    # Agent and obstacle(agent2) control / agent 和 obstacle(agent2) 控制
    # ======================= 
    def step(self, action):
        
        # =======================
        # agent control
        # =======================
        # 关节索引范围（8 个）
        joint_ids = np.arange(0, 8)
        actutator_ids = np.arange(0, 8)
        action_ids = np.arange(0, 8)

        # 当前状态
        qpos_now = self.data.qpos[joint_ids]
        qvel_now = self.data.qvel[joint_ids]
        
        action[action_ids] = np.clip(action[action_ids], -1, 1)  # 限制到[-1,1]
        qpos_target = self.action_to_qpos(action[action_ids],joint_ids)
        qvel_target = 0.0

        
        # PD 控制扭矩
        torque = self.Kp1 * (qpos_target - qpos_now) + self.Kd1 * (qvel_target - qvel_now)

        # 限幅
        ctrl_min = self.model.actuator_ctrlrange[actutator_ids, 0]
        ctrl_max = self.model.actuator_ctrlrange[actutator_ids, 1]
        self.data.ctrl[actutator_ids] = np.clip(torque, ctrl_min, ctrl_max)

        # =======================
        # obstacle(agent_2) control
        # =======================
        # 关节索引范围（8个）
        joint_ids = np.arange(12, 20)
        actutator_ids = np.arange(8, 16)
        action_ids = np.arange(8, 16)
        
        # 当前状态
        qpos_now = self.data.qpos[joint_ids]
        qvel_now = self.data.qvel[joint_ids]
        
        action[action_ids] = np.clip(action[action_ids], -1, 1)  # 限制到[-1,1]
        qpos_target = self.action_to_qpos(action[action_ids],joint_ids)
        qvel_target = 0.0

        
        # PD 控制扭矩
        torque = self.Kp2 * (qpos_target - qpos_now) + self.Kd2 * (qvel_target - qvel_now)

        # 限幅
        ctrl_min = self.model.actuator_ctrlrange[actutator_ids, 0]
        ctrl_max = self.model.actuator_ctrlrange[actutator_ids, 1]
        self.data.ctrl[actutator_ids] = np.clip(torque, ctrl_min, ctrl_max)
                              
                
        # =======================
        # Simulation one step / 仿真一步
        # ======================= 
        mujoco.mj_step(self.model, self.data)
        self.time += self.dt
        self.episode_step += 1

        # =======================
        # Render mode switching
        # =======================
        if self.render_mode == "human":
            self.render()
        elif self.render_mode == "rgb_array":
            frame = self.render()
            info["frame"] = frame
    
        # =======================
        # If the viewer is enabled, synchronize the screen / 如果开启了 viewer ，则同步画面
        # =======================
        if self.viewer is not None:
            self.viewer.sync()
        
        # =======================
        # Constructing observation space / 构造观测空间
        # ======================= 
        obs = np.concatenate([self.data.qpos[:8], 
                              self.data.qvel[:8], 
                              self.data.qpos[12:20],
                              
                              self.data.qpos[12:20], 
                              self.data.qvel[12:20], 
                              self.data.qpos[:8]])

        # =======================
        # Constructing reward function / 构造奖励函数
        # =======================
        #1. Long distance punishment/close range reward
        #(Encourage moving towards the target)
        # 1、远距离惩罚/近距离奖励
        #（鼓励向目标移动）
        
        # agent
        agent_ee_body_id = self.model.body("tip8_agent").id
        agent_ee_pos = self.data.xpos[agent_ee_body_id]
        obstacle_base_body_id = self.model.body("base_link_obstacle").id
        target_pos =  self.data.xpos[obstacle_base_body_id]
        dist_to_target = np.linalg.norm(agent_ee_pos - target_pos)
        
        R_dist = dist_to_target**2
        w_dist = -1.0
            
        # obstacle(agent2)
        obstacle_ee_body_id = self.model.body("tip8_obstacle").id
        obstacle_ee_pos = self.data.xpos[obstacle_ee_body_id]
        agent_base_body_id = self.model.body("base_link_agent").id
        target_pos2 =  self.data.xpos[agent_base_body_id]
        dist_to_target2 = np.linalg.norm(obstacle_ee_pos - target_pos2)
        
        R_dist2 = dist_to_target2**2
        w_dist2 = -1.0

        # 2、Reward for Success and Punishment for Failure
        # 2、成功奖励与失败惩罚
        ccp = self._check_collision_point ()
        
        # agent
        R_success =  1.0 if ccp == 2 or ccp == 3 else 0.0
        w_success = 10_000.0
        R_failure =  1.0 if ccp == 1 or ccp == 3 else 0.0
        w_failure = -10_000.0
        # obstacle(agent2)
        R_success2 =  1.0 if ccp == 1 or ccp == 3 else 0.0
        w_success2 = 10_000.0
        R_failure2 =  1.0 if ccp == 2 or ccp == 3 else 0.0
        w_failure2 = -10_000.0
        
        # 3. Collision rewards or punishments
        # (It is currently unclear whether collisions should be punished or rewarded, so w_collision is set to 0.0)
        # 3、碰撞奖励或惩罚
        # （暂时不清楚是否应该对碰撞进行惩罚或奖励，因此将 w_collision 设置为 0.0）
        ccnp = self._check_collision_no_point()
        
        # agent
        R_collision = 1.0 if ccnp else 0.0
        w_collision = 0.0
        # obstacle(agent2)
        R_collision2 = 1.0 if ccnp else 0.0
        w_collision2 = 0.0
        
        # 4、动作平滑度惩罚 或 控制开销惩罚
        #（鼓励更平滑的路径）
        # 4.Punishment for smoothness of actions
        # (Encourage smoother paths)
        # agent
        R_action = np.sum(np.square(action[0:8]))
        w_action = -0.01
        # obstacle(agent2)
        R_action2 = np.sum(np.square(action[8:16]))
        w_action2 = -0.01
        
        # 5. Time step punishment 
        # (encouraging efficiency)
        # 5、时间步惩罚
        # （鼓励效率）
        # agent
        R_step = 1
        w_step = -0.1
        # obstacle(agent2)
        R_step2 = 1
        w_step2 = -0.1
        
        # 构建分项的奖励函数信息数组
        R_separate = np.array([
            w_dist * R_dist,
            w_success * R_success,
            w_failure * R_failure,
            w_collision * R_collision,
            w_action * R_action,
            w_step * R_step,
            
            w_dist2 * R_dist2,
            w_success2 * R_success2,
            w_failure2 * R_failure2,
            w_collision2 * R_collision2,
            w_action2 * R_action2,
            w_step2 * R_step2,   
        ])
        
        # Calculate the reward function value / 计算奖励函数值
        # agent
        reward1 = w_dist * R_dist + w_success * R_success + w_failure * R_failure +\
                 w_collision * R_collision + w_action * R_action + w_step * R_step
        # obstacle(agent2)
        reward2 = w_dist2 * R_dist2 + w_success2 * R_success2 + w_failure2 * R_failure2 +\
                 w_collision2 * R_collision2 + w_action2 * R_action2 + w_step2 * R_step2
        reward = np.array([reward1,reward2])

        # =======================
        # End flag / 结束标志
        # ======================= 
        
        terminated = self._check_collision_point() != 0
        truncated = self.episode_step >= self.max_episode_steps
        info = {
            "R_separate": R_separate
            }
        
        # =======================
        # Return / 返回
        # =======================
        return obs, reward, terminated, truncated, info
    
    # =======================
    # Render mode switching / 渲染环境切换
    # ======================= 
    def render(self):
        if self.render_mode == 'human':
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        elif self.render_mode == 'rgb_array':
            # If necessary, implement off screen rendering / 如果需要，实现离屏渲染
            pass
    
    # =======================sdddd
    # Reset / 重置
    # =======================   
    def reset(self, seed=None, options=None):  
        super().reset(seed=seed)        
        # Reset simulation / 重置模拟状态
        mujoco.mj_resetData(self.model, self.data) 
        self.time = 0.0
        self.episode_step = 0
        # Constructing observation space / 构造观测空间
        
        obs = np.concatenate([self.data.qpos[:8], 
                              self.data.qvel[:8], 
                              self.data.qpos[12:20],
                              self.data.qpos[12:20], 
                              self.data.qvel[12:20], 
                              self.data.qpos[:8]])
        info = {} 
        # Return / 返回
        return obs, info  

    # =======================
    # Release resources / 释放资源
    # ======================= 
    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
            print("🛑 Close MuJoCo viewer / 关闭 MuJoCo viewer")
    
    
