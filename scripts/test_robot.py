import numpy as np
import pybullet as p
from environments.robot_env import RobotEnv
from models.td3 import TD3, ActorNetwork, CriticNetwork
import torch

def test_robot_with_trained_model(total_steps=5000, render=True, slow_down_factor=2.0):
    """使用预训练模型测试机器人环境的函数，可视化机器人运动"""
    # 创建环境实例
    env = RobotEnv(render=render)

    # 创建代理并加载预训练模型
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    agent = TD3(obs_dim, action_dim)
    agent.actor.load_state_dict(torch.load('models/'+'best_actor_ts171885_r1744.0.pth'))
    agent.critic1.load_state_dict(torch.load('models/'+'best_critic1_ts171885_r1744.0.pth'))
    agent.critic2.load_state_dict(torch.load('models/'+'best_critic2_ts171885_r1744.0.pth'))
    print("加载预训练模型成功！")

    # 记录初始位置和总距离
    start_position, _ = p.getBasePositionAndOrientation(env.robotId)
    total_distance = 0.0
    episode = 0
    episode_distances = []
    episode_rewards = []

    # 重置环境
    observation = env.reset()

    current_episode_reward = 0.0

    # 执行动作并观察结果
    for step in range(total_steps):
        # 使用训练好的模型选择动作（评估时不添加噪声）
        action = agent.select_action(observation, add_noise=False, evaluate=True)

        # 执行动作
        observation, reward, done, info = env.step(action)

        current_episode_reward += reward

        # 如果机器人摔倒，重置环境并记录位置
        if done:
            # 1. 先判断是否“因到达目标区域”导致 done
            pos_x, pos_y, _ = p.getBasePositionAndOrientation(env.robotId)[0]
            dx = pos_x - env.target_x
            dy = pos_y - env.target_y
            dist2d = np.sqrt(dx*dx + dy*dy)

            # 2. 如果的确“到达目标”，则额外加一次性 terminal_bonus
            if dist2d < env.target_threshold:
                print(f"触发目标终局奖励: +{env.terminal_bonus:.1f}")
                current_episode_reward += env.terminal_bonus


            position, _ = p.getBasePositionAndOrientation(env.robotId)
            episode_distance = position[0] - start_position[0]
            episode_distances.append(episode_distance)
            total_distance += episode_distance
            episode_rewards.append(current_episode_reward)

            print(f"\n===== 回合 {episode+1} 结束 =====")
            print(f"在步骤 {step} 摔倒")
            print(f"最终位置: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
            print(f"本回合行走距离: {episode_distance:.3f}米")
            print(f"本回合总奖励: {current_episode_reward:.3f}") 
            print(f"总行走距离: {total_distance:.3f}米\n")
            
            episode += 1
            observation = env.reset()
            start_position, _ = p.getBasePositionAndOrientation(env.robotId)
            current_episode_reward = 0.0

        # 控制模拟速度
        if render and slow_down_factor > 0:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
            import time
            time.sleep(1.0/240.0 * slow_down_factor)

    # 最后一个回合可能没有摔倒，记录最后一回合的奖励
    if not done:
        position, _ = p.getBasePositionAndOrientation(env.robotId)
        episode_distance = position[0] - start_position[0]
        episode_distances.append(episode_distance)
        total_distance += episode_distance
        episode_rewards.append(current_episode_reward)
    
    print(f"\n===== 测试完成 =====")
    print(f"最终位置: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
    print(f"最后一回合行走距离: {episode_distance:.3f}米")
    print(f"总奖励: {current_episode_reward:.3f}")
    print(f"总行走距离: {total_distance:.3f}米")
    
    if len(episode_distances) > 0:
        avg_distance = sum(episode_distances) / len(episode_distances)
        print(f"平均每回合行走距离: {avg_distance:.3f}米")

    # 关闭环境
    env.close()
    print("测试完成")

if __name__ == "__main__":
    test_robot_with_trained_model(render=True, slow_down_factor=1.0)  # 速度减半，便于观察