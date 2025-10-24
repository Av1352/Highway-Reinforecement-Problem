# utils/training.py

import torch
import numpy as np
from tqdm import trange

def train_agent(agent, env, episodes=100, max_steps=1000, visualize=False, save_path=None):
    all_rewards = []
    for ep in trange(episodes, desc="Training episodes"):
        state, _ = env.reset()
        if isinstance(state, tuple) or isinstance(state, list):
            state = np.array(state).flatten()
        total_reward = 0
        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            if isinstance(next_state, tuple) or isinstance(next_state, list):
                next_state = np.array(next_state).flatten()
            agent.memory.push(state, action, reward, next_state, done)
            agent.learn()
            state = next_state
            total_reward += reward
            if done or truncated:
                break
        agent.update_target()
        all_rewards.append(total_reward)
        if visualize:
            print(f"Episode {ep+1}: Reward = {total_reward}")
    if save_path:
        np.savetxt(save_path, all_rewards, delimiter=",")
    return all_rewards

def evaluate_agent(agent, env, episodes=10, max_steps=1000):
    all_rewards = []
    for ep in range(episodes):
        state, _ = env.reset()
        if isinstance(state, tuple) or isinstance(state, list):
            state = np.array(state).flatten()
        total_reward = 0
        for step in range(max_steps):
            action = agent.act(state, epsilon=0.0) # greedy eval
            next_state, reward, done, truncated, _ = env.step(action)
            if isinstance(next_state, tuple) or isinstance(next_state, list):
                next_state = np.array(next_state).flatten()
            state = next_state
            total_reward += reward
            if done or truncated:
                break
        all_rewards.append(total_reward)
    return all_rewards

def train_decision_transformer(model, dataloader, optimizer, criterion, n_epochs):
    ep_bar = trange(n_epochs, desc="DT EPOCHS")
    model.train()
    losses = []
    for epoch in ep_bar:
        epoch_loss = 0
        for s, a, r, t in dataloader:
            optimizer.zero_grad()
            a_preds = model(s, a, r, t)
            a_preds = a_preds.view(-1, model.config.act_dim)
            a = a.view(-1).to(torch.long)
            loss = criterion(a_preds, a)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        mean_loss = epoch_loss / len(dataloader)
        losses.append(mean_loss)
        print(f"Epoch {epoch+1} Loss: {mean_loss}")
    return losses
