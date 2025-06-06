# TD3代理（使用高斯噪声 + 双Critic + 延迟Actor更新 + target smoothing）
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque, namedtuple
import random
import os
import pybullet as p

# 经验回放缓冲区
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        self.memory.append(Experience(*args))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        return [self.memory[i] for i in indices]
        # return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# 演员网络（策略网络）
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.LayerNorm(hidden_dim), # 归一化
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 将动作限制在[-1, 1]范围内
        )

    def forward(self, state):
        return self.network(state)

# 评论家网络（价值网络）
class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512):
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            # nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)
class TD3:
    def __init__(self, obs_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, tau=0.005, buffer_size=1000000, batch_size=512,
                 policy_noise=0.2, noise_clip=0.5, policy_delay=2):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.total_it = 0  # 计步器，判断是否延迟更新Actor

        # 网络
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic1 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.critic2 = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.critic1_target = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.critic2_target = CriticNetwork(obs_dim, action_dim).to(self.device)

        # 同步target
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=lr_critic)

        # 回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, add_noise=True, evaluate=False):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        self.actor.train()

        if add_noise and not evaluate:
            noise = np.random.normal(0, 0.1, size=action.shape)
            action += noise

        return np.clip(action, -1, 1)

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        self.total_it += 1

        # Sample batch
        experiences = self.replay_buffer.sample(self.batch_size)
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences]).reshape(-1, 1)
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences]).reshape(-1, 1)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Target action smoothing
        noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
        next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)

        # Compute target Q
        target_q1 = self.critic1_target(next_states, next_actions)
        target_q2 = self.critic2_target(next_states, next_actions)
        target_q = torch.min(target_q1, target_q2)
        # target_q = rewards + (1 - dones) * self.gamma * target_q
        # 修正后 - 正确的Bellman方程
        target_q = rewards + (1 - dones) * self.gamma * target_q.detach()

        # Update Critic 1
        current_q1 = self.critic1(states, actions)
        critic1_loss = nn.MSELoss()(current_q1, target_q.detach())

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        # Update Critic 2
        current_q2 = self.critic2(states, actions)
        critic2_loss = nn.MSELoss()(current_q2, target_q.detach())

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Delayed Actor updates
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic1(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Soft update target networks
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

# 训练函数（改好版）
def train_td3(env, total_timesteps=1000000, load_actor_path=None, load_critic1_path=None, load_critic2_path=None):
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    agent = TD3(obs_dim, action_dim,
                 lr_actor=3e-4,
                 lr_critic=1e-3,
                 batch_size=256)
    
    # 加载预训练模型（如果提供）
    if load_actor_path and os.path.isfile(load_actor_path):
        agent.actor.load_state_dict(torch.load(load_actor_path, map_location=agent.device))
        agent.actor_target.load_state_dict(torch.load(load_actor_path, map_location=agent.device))
        print(f"Loaded actor parameters from {load_actor_path}")
    
    if load_critic1_path and os.path.isfile(load_critic1_path):
        agent.critic1.load_state_dict(torch.load(load_critic1_path, map_location=agent.device))
        agent.critic1_target.load_state_dict(torch.load(load_critic1_path, map_location=agent.device))
        print(f"Loaded critic1 parameters from {load_critic1_path}")
    
    if load_critic2_path and os.path.isfile(load_critic2_path):
        agent.critic2.load_state_dict(torch.load(load_critic2_path, map_location=agent.device))
        agent.critic2_target.load_state_dict(torch.load(load_critic2_path, map_location=agent.device))
        print(f"Loaded critic2 parameters from {load_critic2_path}")
    
    if not (load_actor_path or load_critic1_path or load_critic2_path):
        print("No pre-trained model found, starting training from scratch.")

    # 训练统计
    episode_rewards = deque(maxlen=100)        # 最近100个回合的奖励
    episode_lengths = deque(maxlen=100)       # 最近100个回合的长度
    timestep = 0
    episode = 0
    best_avg_reward = -float('inf')           # 记录最佳平均奖励
    
    # 重置环境
    state = env.reset()
    episode_reward = 0
    episode_steps = 0

    # 设置保存目录和文件
    save_dir = "trained_models"
    os.makedirs(save_dir, exist_ok=True)
    reward_log_path = os.path.join(save_dir, "training_rewards.csv")
    
    # 创建/初始化奖励记录文件
    if not os.path.exists(reward_log_path):
        with open(reward_log_path, 'w') as f:
            f.write("timestep,avg_reward\n")
    
    # 主训练循环
    while timestep < total_timesteps:
        # 选择和执行动作
        action = agent.select_action(state, add_noise=(timestep < total_timesteps * 0.8))  # 后期减少噪声
        next_state, reward, done, _ = env.step(action)

        # 终局奖励（如果到达目标）
        if done:
            # 读取当前在物理仿真中的位置
            pos, _ = p.getBasePositionAndOrientation(env.robotId)
            pos_x, pos_y = pos[0], pos[1]
            dx = pos_x - env.target_x
            dy = pos_y - env.target_y
            dist2d = np.sqrt(dx*dx + dy*dy)
            if dist2d < env.target_threshold:
                # 只有在“进入目标区域”时才发这笔 bonus
                reward += env.terminal_bonus

        # 存储经验转换
        agent.store_transition(state, action, reward, next_state, done)
        
        # 更新状态和计数器
        state = next_state
        timestep += 1
        episode_reward += reward
        episode_steps += 1

        # 优化代理（更新网络权重）
        if len(agent.replay_buffer) > agent.batch_size:  # 确保有足够经验
            agent.update()

        # 保存和记录（每5000步或终局）
        if timestep % 5000 == 0 or done:
            # 计算平均奖励
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            
            # 保存最佳模型
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
                best_timestep = timestep
                torch.save(agent.actor.state_dict(), os.path.join(save_dir, f"best_actor_ts{timestep}_r{avg_reward:.1f}.pth"))
                torch.save(agent.critic1.state_dict(), os.path.join(save_dir, f"best_critic1_ts{timestep}_r{avg_reward:.1f}.pth"))
                torch.save(agent.critic2.state_dict(), os.path.join(save_dir, f"best_critic2_ts{timestep}_r{avg_reward:.1f}.pth"))
                print(f"🚀 New best model saved at ts {timestep} with avg reward {avg_reward:.1f}!")
            
            # 定期完整保存
            if timestep % 5000 == 0:
                actor_path = os.path.join(save_dir, f"actor_ts{timestep}_r{avg_reward:.1f}.pth")
                critic1_path = os.path.join(save_dir, f"critic1_ts{timestep}_r{avg_reward:.1f}.pth")
                critic2_path = os.path.join(save_dir, f"critic2_ts{timestep}_r{avg_reward:.1f}.pth")
                torch.save(agent.actor.state_dict(), actor_path)
                torch.save(agent.critic1.state_dict(), critic1_path)
                torch.save(agent.critic2.state_dict(), critic2_path)
            
            # 记录训练进度
            with open(reward_log_path, 'a') as f:
                f.write(f"{timestep},{avg_reward:.6f}\n")
        
        # 处理回合结束
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            avg_length = np.mean(episode_lengths) if episode_lengths else 0
            
            # 打印状态
            print(f"🌀 Episode {episode} | "
                  f"Steps: {episode_steps} | "
                  f"Reward: {episode_reward:.1f} | "
                  f"Avg: {avg_reward:.1f} | "
                  f"Length: {avg_length:.1f}")
            
            # 重置回合
            state = env.reset()
            episode_reward = 0
            episode_steps = 0
            episode += 1

    # 保存最终模型
    final_path = os.path.join(save_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    torch.save(agent.actor.state_dict(), os.path.join(final_path, "final_actor.pth"))
    torch.save(agent.critic1.state_dict(), os.path.join(final_path, "final_critic1.pth"))
    torch.save(agent.critic2.state_dict(), os.path.join(final_path, "final_critic2.pth"))
    
    print(f"✅ Training complete! Final models saved in {final_path}")
    return agent

    # 评估函数（不添加噪声）
def evaluate_td3(env, agent, episodes=10):
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            # 评估时不添加噪声
            action = agent.select_action(state, add_noise=False, evaluate=True)
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()  # 可视化评估过程
        total_rewards += episode_reward
        print(f"Evaluation Episode {episode+1}, Reward: {episode_reward:.2f}")
    
    avg_reward = total_rewards / episodes
    print(f"Average Evaluation Reward: {avg_reward:.2f}")
    return avg_reward
