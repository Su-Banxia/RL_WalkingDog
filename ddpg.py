import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from collections import deque, namedtuple
import random
import os

# 经验回放缓冲区
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, *args):
        self.memory.append(Experience(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# 演员网络（策略网络）
class ActorNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512): # TODO:
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # 将动作限制在[-1, 1]范围内
        )
    
    def forward(self, state):
        return self.network(state)

# 评论家网络（价值网络）
class CriticNetwork(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=512): # TODO:
        super(CriticNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)

# DDPG代理（使用高斯噪声）
class DDPG:
    def __init__(self, obs_dim, action_dim, lr_actor=5e-4, lr_critic=5e-3, 
                 gamma=0.95, tau=0.001, buffer_size=1000000, batch_size=256,
                 noise_scale=0.5, noise_decay=0.9995, min_noise_scale=0.3):   # TODO  
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.noise_scale = noise_scale  # 高斯噪声标准差
        self.noise_decay = noise_decay  # 噪声衰减因子
        self.current_noise_scale = noise_scale
        self.min_noise_scale = min_noise_scale
        
        # 创建网络
        self.actor = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.actor_target = ActorNetwork(obs_dim, action_dim).to(self.device)
        self.critic = CriticNetwork(obs_dim, action_dim).to(self.device)
        self.critic_target = CriticNetwork(obs_dim, action_dim).to(self.device)
        
        # 复制权重到目标网络
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # 经验回放缓冲区
        self.replay_buffer = ReplayBuffer(buffer_size)

    def select_action(self, state, add_noise=True, evaluate=False):
        # state = torch.FloatTensor(state.reshape(1, -1))
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().detach().numpy().flatten()
        self.actor.train()
        
        # 在训练时添加噪声，评估时不添加噪声
        if add_noise and not evaluate:
            noise = np.random.normal(0, self.current_noise_scale, size=action.shape)
            action += noise
        
        return np.clip(action, -1, 1)  # 确保动作在[-1, 1]范围内
    
    def decay_noise(self):
        # 在每个Episode结束时衰减噪声
        self.current_noise_scale = max(
            self.current_noise_scale * self.noise_decay,
            self.min_noise_scale
        )

    def update(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # 从回放缓冲区采样
        experiences = self.replay_buffer.sample(self.batch_size)

        # 先将经验转换为numpy数组，再转换为torch张量
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
     
        # 更新评论家网络
        next_actions = self.actor_target(next_states)
        target_q = self.critic_target(next_states, next_actions)
        target_q = rewards + (1 - dones) * self.gamma * target_q
        
        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 更新演员网络
        actor_loss = -self.critic(states, self.actor(states)).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 软更新目标网络
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

# 训练函数
def train_ddpg(env, total_timesteps=1000000, load_actor_path=None, load_critic_path=None):
    obs_dim = env.obs_dim
    action_dim = env.action_dim
    agent = DDPG(obs_dim, action_dim)

    # —— 如果指定了已有模型路径且文件存在，就加载参数 —— #
    if load_actor_path is not None and os.path.isfile(load_actor_path):
        agent.actor.load_state_dict(torch.load(load_actor_path, map_location=agent.device))
        agent.actor_target.load_state_dict(torch.load(load_actor_path, map_location=agent.device))
        print(f"Loaded actor parameters from {load_actor_path}")
    if load_critic_path is not None and os.path.isfile(load_critic_path):
        agent.critic.load_state_dict(torch.load(load_critic_path, map_location=agent.device))
        agent.critic_target.load_state_dict(torch.load(load_critic_path, map_location=agent.device))
        print(f"Loaded critic parameters from {load_critic_path}")
    if load_actor_path is None and load_critic_path is None:
        print("No pre-trained model found, starting training from scratch.")

    
    episode_rewards = deque(maxlen=100)
    timestep = 0
    episode = 0
    
    state = env.reset()

    episode_reward = 0
    
    while timestep < total_timesteps:
        # if episode < 3500:
        #     env.set_phase(2)
        # elif 3500 <= episode < 6000:
        #     env.set_phase(2)
        # else:
        #     env.set_phase(2)
        env.set_phase(2)
        # 选择和执行动作
        action = agent.select_action(state)
        next_state, reward, done, _ = env.step(action)
        
        # 存储转换
        agent.store_transition(state, action, reward, next_state, done)
        
        # 更新状态和计数器
        state = next_state
        timestep += 1
        episode_reward += reward
        
        # 优化代理
        agent.update()
        
        # 处理episode结束
        if done:
            # 记录奖励
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards) if episode_rewards else 0
            
            print(f"Episode {episode}, timestep {timestep}, "
                  f"reward: {episode_reward:.2f}, "
                  f"avg reward: {avg_reward:.2f}, "
                  f"noise_scale: {agent.current_noise_scale:.4f}")
            
            agent.decay_noise()  # 衰减噪声

            # 重置episode计数器
            state = env.reset()
            episode_reward = 0
            episode += 1
    
    # 保存训练好的模型
    torch.save(agent.actor.state_dict(), 'robot_walking_ddpg_actor.pth')
    torch.save(agent.critic.state_dict(), 'robot_walking_ddpg_critic.pth')
    return agent

# 评估函数（不添加噪声）
def evaluate_ddpg(env, agent, episodes=10):
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