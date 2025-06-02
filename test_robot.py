import numpy as np
import pybullet as p
from robot_env import RobotEnv
from ddpg import DDPG, ActorNetwork, CriticNetwork
import torch

def test_robot_with_trained_model(total_steps=2000, render=True, slow_down_factor=2.0):
    """使用预训练模型测试机器人环境的函数，可视化机器人运动"""
    # 创建环境实例
    env = RobotEnv(render=render)

    # 设置奖励权重
    env.set_reward_weights(0.005, 0.5)

    # 创建代理并加载预训练模型
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    agent = DDPG(obs_dim, action_dim)
    agent.actor.load_state_dict(torch.load('robot_walking_ddpg_actor.pth'))
    agent.critic.load_state_dict(torch.load('robot_walking_ddpg_critic.pth'))
    print("加载预训练模型成功！")

    # 重置环境
    observation = env.reset()

    # 执行动作并观察结果
    for step in range(total_steps):
        # 使用训练好的模型选择动作（评估时不添加噪声）
        action = agent.select_action(observation, add_noise=False, evaluate=True)

        # 执行动作
        observation, reward, done, info = env.step(action)

        # 打印简要信息（每100步一次）
        if step % 100 == 0:
            print(f"步骤: {step}, 奖励: {reward:.4f}")
            position, _ = p.getBasePositionAndOrientation(env.robotId)
            print(f"躯干高度: {position[2]:.3f}米")

        # 如果机器人摔倒，重置环境
        if done:
            print(f"机器人在步骤 {step} 摔倒，重置环境")
            observation = env.reset()

        # 控制模拟速度
        if render and slow_down_factor > 0:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
            import time
            time.sleep(1.0/240.0 * slow_down_factor)

    # 关闭环境
    env.close()
    print("测试完成")

if __name__ == "__main__":
    test_robot_with_trained_model(render=True, slow_down_factor=1.0)  # 速度减半，便于观察