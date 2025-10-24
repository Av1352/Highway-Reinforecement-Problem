# agents/a3c_agent.py

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym
import os

# -------- Neural Net (Actor-Critic) --------
class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(np.prod(input_shape), 128),
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32).flatten()
        x = self.fc(x)
        return self.policy(x), self.value(x)

# -------- Agent --------
class Agent:
    def __init__(self, global_model, input_shape, n_actions, gamma, lr, name, global_ep_idx, res_queue, env_id):
        self.local_model = ActorCritic(input_shape, n_actions)
        self.global_model = global_model
        self.optimizer = optim.Adam(self.global_model.parameters(), lr=lr)
        self.gamma = gamma
        self.name = name
        self.global_ep_idx = global_ep_idx
        self.res_queue = res_queue
        self.env = gym.make(env_id)
        self.n_actions = n_actions

    def act(self, state):
        logits, _ = self.local_model(state)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.sample().item()

    def compute_returns(self, rewards, done, next_value):
        R = next_value
        returns = []
        for r in reversed(rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def learn(self, states, actions, rewards, next_state, done):
        states = torch.tensor(np.vstack(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        # Compute local loss
        logits, values = self.local_model(states)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)

        with torch.no_grad():
            _, next_value = self.local_model(next_state)
        returns = self.compute_returns(rewards, done, next_value.item())
        returns = torch.tensor(returns, dtype=torch.float32)
        values = values.squeeze()
        # Advantage
        advantage = returns - values

        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss

        # Update global weights
        self.optimizer.zero_grad()
        loss.backward()
        # Ensure global_model gets grads:
        for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
            global_param._grad = local_param.grad
        self.optimizer.step()
        # Sync local with global
        self.local_model.load_state_dict(self.global_model.state_dict())

# -------- Multiprocessing Training Loop (for advanced/full training) --------
def worker(global_model, input_shape, n_actions, gamma, lr, name, global_ep_idx, res_queue, env_id, max_episode=1000):
    agent = Agent(global_model, input_shape, n_actions, gamma, lr, name, global_ep_idx, res_queue, env_id)
    for episode in range(max_episode):
        state, _ = agent.env.reset()
        done = False
        rewards, states, actions = [], [], []
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = agent.env.step(action)
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            state = next_state
            total_reward += reward
            if done or truncated:
                break
        agent.learn(states, actions, rewards, next_state, done)
        with agent.global_ep_idx.get_lock():
            agent.global_ep_idx.value += 1
        agent.res_queue.put(total_reward)
    agent.res_queue.put(None)

# -------- API for Demo Script --------
def run_demo(env_id="highway-v0", episodes=20):
    env = gym.make(env_id)
    input_shape = env.observation_space.shape
    n_actions = env.action_space.n
    global_model = ActorCritic(input_shape, n_actions)
    gamma = 0.99
    lr = 1e-4
    manager = mp.Manager()
    global_ep_idx = manager.Value('i', 0)
    res_queue = manager.Queue()
    agent = Agent(global_model, input_shape, n_actions, gamma, lr, "demo", global_ep_idx, res_queue, env_id)

    rewards = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        states, actions, rewards_batch = [], [], []
        total_reward = 0
        while not done:
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            states.append(state)
            actions.append(action)
            rewards_batch.append(reward)
            state = next_state
            total_reward += reward
            if done or truncated:
                break
        agent.learn(states, actions, rewards_batch, next_state, done)
        rewards.append(total_reward)
    return rewards

# MAIN for stand-alone test/demo
if __name__ == '__main__':
    demo_rewards = run_demo()
    print("Demo A3C rewards:", demo_rewards)
