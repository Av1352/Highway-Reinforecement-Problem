import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

class ReplayMemory:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.alpha = alpha
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    def push(self, state, action, reward, next_state, done):
        max_priority = self.priorities.max() if self.memory else 1.0
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = (state, action, reward, next_state, done)
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
    def sample(self, batch_size, beta=0.4):
        priorities = self.priorities if len(self.memory) == self.capacity else self.priorities[:self.position]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.memory), batch_size, p=probabilities)
        samples = [self.memory[i] for i in indices]
        weights = (len(self.memory) * probabilities[indices]) ** (-beta)
        weights /= weights.max()
        return samples, indices, weights
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
    def __len__(self):
        return len(self.memory)

class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.advantage_fc = nn.Linear(128, action_dim)
        self.value_fc = nn.Linear(128, 1)
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        advantage = self.advantage_fc(x)
        value = self.value_fc(x)
        return value + advantage - advantage.mean()

class RainbowAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, batch_size=64, memory_size=100000):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.policy_net = DuelingDQN(state_dim, action_dim).to("cuda")
        self.target_net = DuelingDQN(state_dim, action_dim).to("cuda")
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(memory_size)
        self.update_target()
    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    def act(self, state, epsilon=0.01):
        if random.random() > epsilon:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to("cuda")
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).item()
        else:
            return random.randrange(self.action_dim)
    def compute_loss(self, samples, indices, weights):
        states = np.array([s for s, _, _, _, _ in samples])
        actions = np.array([a for _, a, _, _, _ in samples])
        rewards = np.array([r for _, _, r, _, _ in samples])
        next_states = np.array([ns for _, _, _, ns, _ in samples])
        dones = np.array([d for _, _, _, _, d in samples])
        states = torch.from_numpy(states).float().to("cuda")
        actions = torch.from_numpy(actions).long().to("cuda")
        rewards = torch.from_numpy(rewards).float().to("cuda")
        next_states = torch.from_numpy(next_states).float().to("cuda")
        dones = torch.from_numpy(dones).float().to("cuda")
        weights = torch.tensor(weights, dtype=torch.float32).to("cuda")
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
        next_q = self.target_net(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q
        loss = (weights * (current_q - target_q.detach()).pow(2)).mean()
        return loss
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        samples, indices, weights = self.memory.sample(self.batch_size)
        loss = self.compute_loss(samples, indices, weights)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            states, actions, rewards, next_states, dones = zip(*samples)
            states = torch.tensor(states, dtype=torch.float32).view(self.batch_size, -1).to("cuda")
            actions = torch.tensor(actions).to("cuda")
            next_states = torch.tensor(next_states, dtype=torch.float32).view(self.batch_size, -1).to("cuda")
            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q = self.target_net(next_states).max(1)[0]
            target_q = torch.tensor(rewards, dtype=torch.float32).to("cuda") + \
                       (1 - torch.tensor(dones, dtype=torch.float32).to("cuda")) * self.gamma * next_q
            td_errors = (current_q - target_q).abs().cpu().numpy()
            self.memory.update_priorities(indices, td_errors.tolist())
