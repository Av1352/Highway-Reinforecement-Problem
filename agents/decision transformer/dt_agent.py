import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange
import numpy as np
import gymnasium as gym
import highway_env
import random
import os

# ---- Data Collector ----

def collect_highway_data(output_dir='highway_ppo_data', max_timesteps=2_400_000):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    env = gym.make("highway-v0")
    env.configure({
        "observation": {
            "type": "GrayscaleObservation",
            "weights": [0.2989, 0.5870, 0.1140],
            "stack_size": 4,
        },
        "absolute": True
    })
    env = gym.make('highway-v0', render_mode='rgb_array')
    env = DummyVecEnv([lambda: env])
    os.makedirs(output_dir, exist_ok=True)

    observations = np.zeros((max_timesteps,) + env.observation_space.shape, dtype=np.float32)
    actions = np.zeros((max_timesteps, env.action_space.shape[0]), dtype=env.action_space.dtype)
    rewards = np.zeros(max_timesteps, dtype=np.float32)
    terminated = np.zeros(max_timesteps, dtype=bool)
    model = PPO('CnnPolicy', env, verbose=1)
    obs, _ = env.reset()
    timestep = 0

    while timestep < max_timesteps:
        observations[timestep] = obs[0]
        action, _ = model.predict(obs)
        obs, reward, done, _, info = env.step(action)
        actions[timestep] = action[0]
        rewards[timestep] = reward[0]
        terminated[timestep] = done[0]
        timestep += 1
        if done[0]:
            obs, _ = env.reset()
        if timestep % 100_000 == 0:
            np.save(os.path.join(output_dir, f'observations_{timestep}.npy'), observations[:timestep])
            np.save(os.path.join(output_dir, f'actions_{timestep}.npy'), actions[:timestep])
            np.save(os.path.join(output_dir, f'rewards_{timestep}.npy'), rewards[:timestep])
            np.save(os.path.join(output_dir, f'terminated_{timestep}.npy'), terminated[:timestep])
    np.save(os.path.join(output_dir, 'observations.npy'), observations[:timestep])
    np.save(os.path.join(output_dir, 'actions.npy'), actions[:timestep])
    np.save(os.path.join(output_dir, 'rewards.npy'), rewards[:timestep])
    np.save(os.path.join(output_dir, 'terminated.npy'), terminated[:timestep])
    env.close()

# ---- Data Loader ----
class RLDataProcessor:
    def __init__(self):
        self.file_path = None
        self.actions = None
        self.observations = None
        self.rewards = None
        self.terminals = None
        self.returns = None
        self.timesteps = None

    def load_data(self, file_path):
        self.file_path = file_path
        self.actions = np.load(f"{file_path}/actions.npy", allow_pickle=True)
        self.observations = np.load(f"{file_path}/observations.npy", allow_pickle=True)
        self.rewards = np.load(f"{file_path}/rewards.npy", allow_pickle=True)
        self.terminals = np.load(f"{file_path}/terminated.npy", allow_pickle=True)

    def calculate_returns_and_timesteps(self):
        self.returns = np.zeros(len(self.rewards))
        self.timesteps = np.zeros(len(self.rewards))
        terminal_indices = np.where(self.terminals == 1)[0]
        last_terminal_index = np.max(np.where(self.terminals == 1))
        for episode_end in reversed(terminal_indices):
            cumulative_return = 0
            current_timestep = 1
            episode_idx = np.where(terminal_indices == episode_end)[0][0]
            start_index = 0 if episode_idx == 0 else terminal_indices[episode_idx - 1] + 1
            for i in range(episode_end, start_index - 1, -1):
                cumulative_return += self.rewards[i]
                self.returns[i] = cumulative_return
                self.timesteps[i] = current_timestep
                current_timestep += 1
        if last_terminal_index + 1 < len(self.rewards):
            self.returns[last_terminal_index + 1:] = 0
            self.timesteps[last_terminal_index + 1:] = 0
            self.actions[last_terminal_index + 1:] = 0
            self.observations[last_terminal_index + 1:] = 0

class RLDataset(Dataset):
    def __init__(self, processor, sequence_len, stack_size=4):
        super().__init__()
        self.sequence_len = sequence_len
        self.stack_size = stack_size
        self.observations = torch.tensor(processor.observations, dtype=torch.float16) / 255
        self.actions = torch.tensor(processor.actions, dtype=torch.int32)
        self.returns = torch.tensor(processor.returns, dtype=torch.float16)
        self.timesteps = torch.tensor(processor.timesteps, dtype=torch.int32)

    def __len__(self):
        return len(self.timesteps) - self.sequence_len - self.stack_size + 1

    def __getitem__(self, idx):
        stacked_obs_seq = torch.stack(
            [torch.stack([self.observations[idx + t + offset] for offset in range(self.stack_size)], dim=0)
             for t in range(self.sequence_len)], dim=0)
        actions_seq = self.actions[idx + self.stack_size - 1: idx + self.stack_size - 1 + self.sequence_len]
        returns_seq = self.returns[idx + self.stack_size - 1: idx + self.stack_size - 1 + self.sequence_len]
        timesteps_seq = self.timesteps[idx + self.stack_size - 1: idx + self.stack_size - 1 + self.sequence_len]
        return stacked_obs_seq, actions_seq, returns_seq, timesteps_seq

# ---- Decision Transformer Model ----
class DecisionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_t = nn.Embedding(config.max_length, config.hidden_dim).to(torch.float16)
        self.embed_a = nn.Embedding(config.act_dim, config.hidden_dim).to(torch.float16)
        self.embed_r = nn.Linear(1, config.hidden_dim).to(torch.float16)
        self.embed_s = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4, padding=0).to(torch.float16),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=0).to(torch.float16),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1, padding=0).to(torch.float16),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, config.hidden_dim).to(torch.float16),
            nn.Tanh()
        )
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.n_heads,
            dim_feedforward=4 * config.hidden_dim,
            dropout=0.1
        ).to(torch.float16)
        self.transformer = nn.TransformerDecoder(
            self.decoder_layer,
            num_layers=config.n_layers
        ).to(torch.float16)
        self.pred_a = nn.Linear(config.hidden_dim, config.act_dim).to(torch.float16)

    def forward(self, s, a, r, t):
        s = s.to(torch.float16)
        r = r.to(torch.float16)
        pos_embedding = self.embed_t(t)
        a_embedding = self.embed_a(a) + pos_embedding
        r_embedding = self.embed_r(r.unsqueeze(-1)) + pos_embedding
        s = s.reshape(-1, 4, 84, 84)
        s_embedding = self.embed_s(s)
        s_embedding = s_embedding.view(2, 10, self.config.hidden_dim) + pos_embedding
        input_embeds = torch.stack((r_embedding, s_embedding, a_embedding), dim=2)
        input_embeds = input_embeds.flatten(1, 2)
        seq_len = input_embeds.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        input_embeds = input_embeds.permute(1, 0, 2)
        hidden_states = self.transformer(input_embeds, input_embeds, tgt_mask=causal_mask)
        hidden_states = hidden_states.permute(1, 0, 2)
        a_hidden = hidden_states[:, 2::3, :]
        return self.pred_a(a_hidden)

# ---- Model config ----

class Config:
    max_length = 2833
    act_dim = 6
    hidden_dim = 192
    n_heads = 3
    n_layers = 4

# ---- Training logic ----

def train_model(model, dataloader, optimizer, criterion, n_epochs):
    ep_bar = trange(n_epochs, desc="epoch bar")
    model.train()
    for epoch in ep_bar:
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}", leave=True)
        for s, a, r, t in batch_bar:
            optimizer.zero_grad()
            a_preds = model(s, a, r, t)
            a_preds = a_preds.view(-1, Config.act_dim)
            a = a.view(-1)
            a = a.to(torch.long)
            loss = criterion(a_preds, a)
            print(loss)
            loss.backward()
            clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            batch_bar.set_postfix({"Loss": loss.item()})

