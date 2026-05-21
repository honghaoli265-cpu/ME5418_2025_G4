# main_test_duel_env.py

"""
Set render mode.
After running, load the environment and simulate several episodes before closing the viewer.
设置渲染模式.
运行后加载环境，仿真重复若干episodes后关闭viewer。
"""

import os
import mujoco
import mujoco.viewer
import numpy as np
import time
from duel_env import DuelEnv

# =======================
# Set for path / 路径设置
# =======================
current_dir = os.path.dirname(__file__)  # Current script file path / 当前脚本路径
xml_path = os.path.join(current_dir, "Fencing_agent&obstacle_description", "fencing_arm_ver3.xml")

# =======================
# Set for steps number & episodes number / 仿真步数、episodes数设置
# =======================
n_steps = 2000   # steps number / 每episode仿真步数
n_episodes = 2   # episodes number / 运行episodes数

# =======================
# Initialization / 初始化
# =======================
render_mode="human"
print(f"Current render mode/ 当前渲染模式{render_mode}")

# Open env
env = DuelEnv(xml_path, render_mode)

model = env.model
data = env.data

# Get joint and actuator names / 获取关节和执行器名称
joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, j) for j in range(model.nu)]

# =======================
# Start simulation episodes / 开始仿真若干episodes
# =======================
for episode in range(n_episodes):
    print(f"🚀 Start the {episode+1}th simulation / 开始第 {episode+1} 次仿真")
    
    obs, info = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
    
    episode_reward = np.array([0.0, 0.0])  # Initialize the cumulative reward for this episode / 初始化该 episode 的累计 reward

    for step in range(n_steps):
        
        action1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # agent
        action2 = np.array([0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]) # obstacle(agent2)
        
        action = np.concatenate((action1, action2))
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        obs1 = obs[:24] # agent
        obs2 = obs[24:] # obstacle(agent2)
        
        reward1 = reward[0] # agent
        reward2 = reward[1] # obstacle(agent2)
        
        episode_reward += reward
        episode_reward1 = episode_reward[0] # agent
        episode_reward2 = episode_reward[1] # obstacle(agent2)
        
        R_separate = info.get("R_separate", None)
        R_seperate1 = R_separate[:6] # agent
        R_seperate2 = R_separate[6:] # obstacle(agent2)
        
        done = terminated or truncated

        if done:
            break
            
    # Print the sum reward after each episode ends / 每个 episode 结束后打印累计 reward
    print(f"🏁 Episode {episode+1} finished, sum reward1: {episode_reward1}, sum reward2: {episode_reward2}")

print("✅ All simulations are completed / 所有仿真完成")

# =======================
# release viewer / 释放窗口
# =======================
env.close()
time.sleep(0.1)

