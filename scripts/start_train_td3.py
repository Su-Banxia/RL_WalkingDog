import sys
import os
import argparse
from environments.robot_env import RobotEnv  
from models.td3 import train_td3, evaluate_td3, TD3  
import torch

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a bipedal robot with TD3')
    parser.add_argument('--train', action='store_true', help='Train the agent')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the trained agent')
    parser.add_argument('--timesteps', type=int, default=1000000, help='Total training timesteps')
    parser.add_argument('--render', action='store_true', help='Render the environment during evaluation')
    parser.add_argument('--load_actor', type=str, default=None, help='Path to pre-trained actor .pth')
    parser.add_argument('--load_critic1', type=str, default=None, help='Path to pre-trained critic1 .pth')
    parser.add_argument('--load_critic2', type=str, default=None, help='Path to pre-trained critic2 .pth')
    args = parser.parse_args()

    # Create robot environment
    env = RobotEnv(render=args.render)

    if args.train:
        # Train the model
        print(f"Starting TD3 agent training, total timesteps: {args.timesteps}")
        agent = train_td3(env, 
                           total_timesteps=args.timesteps,
                           load_actor_path=args.load_actor,
                           load_critic1_path=args.load_critic1,
                           load_critic2_path=args.load_critic2)
        print("Training completed! Models saved as robot_walking_td3_actor.pth, robot_walking_td3_critic1.pth 和 robot_walking_td3_critic2.pth")

    if args.evaluate:
        # Evaluate the model
        try:
            # Create agent and load pretrained models
            obs_dim = env.obs_dim
            action_dim = env.action_dim
            agent = TD3(obs_dim, action_dim)
            
            agent.actor.load_state_dict(torch.load('robot_walking_td3_actor.pth'))
            agent.critic1.load_state_dict(torch.load('robot_walking_td3_critic1.pth'))
            agent.critic2.load_state_dict(torch.load('robot_walking_td3_critic2.pth'))
            print("Successfully loaded pretrained models!")
            
            # Evaluate agent
            print("Starting agent evaluation...")
            evaluate_td3(env, agent, episodes=10)
            
        except FileNotFoundError:
            print("Error: Pretrained models not found. Please train the model first or ensure model files are in the correct path.")

    # Close environment
    env.close()

if __name__ == "__main__":
    main()
