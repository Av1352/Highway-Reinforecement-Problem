# agents/a3c_agent.py

import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gymnasium as gym

class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        self.feature_dim = int(np.prod(input_shape))
        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, 128),
            nn.ReLU(),
        )
        self.policy = nn.Linear(128, n_actions)
        self.value = nn.Linear(128, 1)

    def forward(self, x):
        # Accepts np.ndarray, list, or torch.Tensor; batch or single
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x.astype(np.float32))
        elif isinstance(x, list):
            x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        # Flatten to (batch, features)
        x = x.contiguous().view(x.size(0), -1)
        x = self.fc(x)
        return self.policy(x), self.value(x)

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
        # Guarantee flattened for input
        obs = np.array(state, dtype=np.float32).flatten()
        logits, _ = self.local_model(obs)
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
        # Flatten EVERY state for proper shape [batch, features]
        states_proc = [np.array(s, dtype=np.float32).flatten() for s in states]
        states_tensor = torch.from_numpy(np.vstack(states_proc))
        actions = torch.tensor(actions, dtype=torch.long)
        logits, values = self.local_model(states_tensor)
        dist = torch.distributions.Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        # Flatten next_state
        next_state_proc = np.array(next_state, dtype=np.float32).flatten()
        with torch.no_grad():
            _, next_value = self.local_model(next_state_proc)
        returns = self.compute_returns(rewards, done, next_value.item())
        returns = torch.tensor(returns, dtype=torch.float32)
        values = values.squeeze()
        advantage = returns - values
        actor_loss = -(log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()
        loss = actor_loss + critic_loss
        self.optimizer.zero_grad()
        loss.backward()
        for local_param, global_param in zip(self.local_model.parameters(), self.global_model.parameters()):
            global_param._grad = local_param.grad
        self.optimizer.step()
        self.local_model.load_state_dict(self.global_model.state_dict())

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

if __name__ == '__main__':
    print("RUN DEMO")
    demo_rewards = run_demo()
    print("Demo A3C rewards:", demo_rewards)
