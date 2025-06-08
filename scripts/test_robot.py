import numpy as np
import pybullet as p
from environments.robot_env import RobotEnv
from models.td3 import TD3, ActorNetwork, CriticNetwork
import torch

def test_robot_with_trained_model(total_steps=5000, render=True, slow_down_factor=2.0):
    """Function to test robot environment with pretrained model, visualizing robot movement"""
    # Create environment instance
    env = RobotEnv(render=render)

    # Create agent and load pretrained models
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    agent = TD3(obs_dim, action_dim)
    agent.actor.load_state_dict(torch.load('models/'+'final_actor.pth'))
    agent.critic1.load_state_dict(torch.load('models/'+'final_critic1.pth'))
    agent.critic2.load_state_dict(torch.load('models/'+'final_critic2.pth'))
    print("Successfully loaded pretrained models!")

    # Record initial position and total distance
    start_position, _ = p.getBasePositionAndOrientation(env.robotId)
    total_distance = 0.0
    episode = 0
    episode_distances = []
    episode_rewards = []

    # Reset environment
    observation = env.reset()

    current_episode_reward = 0.0

    # Execute actions and observe results
    for step in range(total_steps):
        # Select action using trained model (no noise during evaluation)
        action = agent.select_action(observation, add_noise=False, evaluate=True)

        # Execute action
        observation, reward, done, info = env.step(action)

        current_episode_reward += reward

        # If robot falls, reset environment and record position
        if done:
            # 1. First check if termination was due to reaching target area
            pos_x, pos_y, _ = p.getBasePositionAndOrientation(env.robotId)[0]
            dx = pos_x - env.target_x
            dy = pos_y - env.target_y
            dist2d = np.sqrt(dx*dx + dy*dy)

            # 2. If indeed reached target, add one-time terminal bonus
            if dist2d < env.target_threshold:
                print(f"Triggered terminal bonus: +{env.terminal_bonus:.1f}")
                current_episode_reward += env.terminal_bonus

            position, _ = p.getBasePositionAndOrientation(env.robotId)
            episode_distance = position[0] - start_position[0]
            episode_distances.append(episode_distance)
            total_distance += episode_distance
            episode_rewards.append(current_episode_reward)

            print(f"\n===== Episode {episode+1} Ended =====")
            print(f"Fallen at step {step}")
            print(f"Final position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
            print(f"Episode distance: {episode_distance:.3f} meters")
            print(f"Episode total reward: {current_episode_reward:.3f}") 
            print(f"Total distance: {total_distance:.3f} meters\n")
            
            episode += 1
            observation = env.reset()
            start_position, _ = p.getBasePositionAndOrientation(env.robotId)
            current_episode_reward = 0.0

        # Control simulation speed
        if render and slow_down_factor > 0:
            p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING)
            import time
            time.sleep(1.0/240.0 * slow_down_factor)

    # Last episode might not have ended, record its reward
    if not done:
        position, _ = p.getBasePositionAndOrientation(env.robotId)
        episode_distance = position[0] - start_position[0]
        episode_distances.append(episode_distance)
        total_distance += episode_distance
        episode_rewards.append(current_episode_reward)
    
    print(f"\n===== Testing Completed =====")
    print(f"Final position: ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})")
    print(f"Last episode distance: {episode_distance:.3f} meters")
    print(f"Total reward: {current_episode_reward:.3f}")
    print(f"Total distance: {total_distance:.3f} meters")
    
    if len(episode_distances) > 0:
        avg_distance = sum(episode_distances) / len(episode_distances)
        print(f"Average episode distance: {avg_distance:.3f} meters")

    # Close environment
    env.close()
    print("Testing completed")

if __name__ == "__main__":
    test_robot_with_trained_model(render=True, slow_down_factor=1.0)  # Half speed for better observation