import numpy as np
import pybullet as p
from robot_env import RobotEnv
from ddpg import DDPG, ActorNetwork, CriticNetwork
import torch

def test_robot_with_trained_model(total_steps=2000, render=True, slow_down_factor=2.0):
    """使用预训练模型测试机器人环境的函数，可视化机器人运动"""
    # 创建环境实例
    env = RobotEnv(render=render)

    # 创建代理并加载预训练模型
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    agent = DDPG(obs_dim, action_dim)
    agent.actor.load_state_dict(torch.load('trained_models/robot_walking_ddpg_actor_40000_r822.53.pth'))
    agent.critic.load_state_dict(torch.load('trained_models/robot_walking_ddpg_critic_40000_r822.53.pth'))
    print("加载预训练模型成功！")

    # 记录初始位置和总距离
    start_position, _ = p.getBasePositionAndOrientation(env.robotId)
    total_distance = 0.0
    episode = 0
    episode_distances = []

    # 重置环境
    observation = env.reset()

    env.set_phase(2)

    # 执行动作并观察结果
    for step in range(total_steps):
        # 使用训练好的模型选择动作（评估时不添加噪声）
        action = agent.select_action(observation, add_noise=False, evaluate=True)

        # 执行动作
        observation, reward, done, info = env.step(action)

        # 打印简要信息（每100步一次）
        if step % 100 == 0:
            position, _ = p.getBasePositionAndOrientation(env.robotId)
            print(f"步骤: {step}, 即时奖励: {reward:.4f}, ", f"躯干高度: {position[2]:.3f}米")

        # # 如果机器人摔倒，重置环境
        # if done:
        #     print(f"机器人在步骤 {step} 摔倒，重置环境")
        #     observation = env.reset()
        # 如果机器人摔倒，重置环境并记录位置
        if done:
            position, _ = p.getBasePositionAndOrientation(env.robotId)
            episode_distance = position[0] - start_position[0]
            episode_distances.append(episode_distance)
            total_distance += episode_distance
            
            print(f"\n===== 回合 {episode+1} 结束 =====")
            print(f"在步骤 {step} 摔倒")
            print(f"最终位置: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
            print(f"本回合行走距离: {episode_distance:.3f}米")
            print(f"总行走距离: {total_distance:.3f}米\n")
            
            episode += 1
            observation = env.reset()
            start_position, _ = p.getBasePositionAndOrientation(env.robotId)

        # 控制模拟速度
        if render and slow_down_factor > 0:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
            import time
            time.sleep(1.0/240.0 * slow_down_factor)

    # 测试结束后，再次获取最终位置
    position, _ = p.getBasePositionAndOrientation(env.robotId)
    episode_distance = position[0] - start_position[0]
    episode_distances.append(episode_distance)
    total_distance += episode_distance
    
    print(f"\n===== 测试完成 =====")
    print(f"最终位置: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
    print(f"最后一回合行走距离: {episode_distance:.3f}米")
    print(f"总行走距离: {total_distance:.3f}米")
    
    if len(episode_distances) > 0:
        avg_distance = sum(episode_distances) / len(episode_distances)
        print(f"平均每回合行走距离: {avg_distance:.3f}米")

    # 关闭环境
    env.close()
    print("测试完成")

if __name__ == "__main__":
    test_robot_with_trained_model(render=True, slow_down_factor=0.5)  # 速度减半，便于观察