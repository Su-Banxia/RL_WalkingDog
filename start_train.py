import sys
import os
import argparse
from robot_env import RobotEnv  # 导入你的机器人环境
from ddpg import train_ddpg, evaluate_ddpg  # 导入DDPG训练和评估函数

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Train a bipedal robot with DDPG')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the trained agent')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total training timesteps')
    parser.add_argument('--render', action='store_true', help='Render the environment during evaluation')
    args = parser.parse_args()

    # 创建机器人环境
    env = RobotEnv(render=args.render)

    if args.train:
        # 训练模型
        print(f"开始训练DDPG代理，总时间步: {args.timesteps}")
        agent = train_ddpg(env, total_timesteps=args.timesteps)
        print("训练完成！模型已保存为 robot_walking_ddpg_actor.pth 和 robot_walking_ddpg_critic.pth")

    if args.evaluate:
        # 评估模型
        try:
            from ddpg import DDPG, ActorNetwork, CriticNetwork
            # 创建代理并加载预训练模型
            obs_dim = env.obs_dim
            action_dim = env.action_dim
            agent = DDPG(obs_dim, action_dim)
            agent.actor.load_state_dict(torch.load('robot_walking_ddpg_actor.pth'))
            agent.critic.load_state_dict(torch.load('robot_walking_ddpg_critic.pth'))
            print("加载预训练模型成功！")
            
            # 评估代理
            print("开始评估代理性能...")
            evaluate_ddpg(env, agent, episodes=10)
            
        except FileNotFoundError:
            print("错误: 找不到预训练模型。请先训练模型或确保模型文件在正确的路径下。")

    # 关闭环境
    env.close()

if __name__ == "__main__":
    main()