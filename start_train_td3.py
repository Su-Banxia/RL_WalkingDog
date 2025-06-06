# start_train_td3.py

import sys
import os
import argparse
from robot_env import RobotEnv  # 导入你的机器人环境
from td3 import train_td3, evaluate_td3, TD3  # 导入td3训练和评估函数
import torch

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train a bipedal robot with TD3')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the trained agent')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total training timesteps')
    parser.add_argument('--render', action='store_true', help='Render the environment during evaluation')
    parser.add_argument('--load_actor', type=str, default=None, help='Path to pre-trained actor .pth')
    parser.add_argument('--load_critic1', type=str, default=None, help='Path to pre-trained critic1 .pth')
    parser.add_argument('--load_critic2', type=str, default=None, help='Path to pre-trained critic2 .pth')
    args = parser.parse_args()

    # 创建机器人环境
    env = RobotEnv(render=args.render)

    if args.train:
        # 训练模型
        print(f"开始训练TD3代理，总时间步: {args.timesteps}")
        agent = train_td3(env, 
                           total_timesteps=args.timesteps,
                           load_actor_path=args.load_actor,
                           load_critic1_path=args.load_critic1,
                           load_critic2_path=args.load_critic2)
        print("训练完成！模型已保存为 robot_walking_td3_actor.pth, robot_walking_td3_critic1.pth 和 robot_walking_td3_critic2.pth")

    if args.evaluate:
        # 评估模型
        try:
            # 创建代理并加载预训练模型
            obs_dim = env.obs_dim
            action_dim = env.action_dim
            agent = TD3(obs_dim, action_dim)
            
            agent.actor.load_state_dict(torch.load('robot_walking_td3_actor.pth'))
            agent.critic1.load_state_dict(torch.load('robot_walking_td3_critic1.pth'))
            agent.critic2.load_state_dict(torch.load('robot_walking_td3_critic2.pth'))
            print("加载预训练模型成功！")
            
            # 评估代理
            print("开始评估代理性能...")
            evaluate_td3(env, agent, episodes=10)
            
        except FileNotFoundError:
            print("错误: 找不到预训练模型。请先训练模型或确保模型文件在正确的路径下。")

    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()
