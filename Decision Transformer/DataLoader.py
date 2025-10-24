from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


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
        self.file_path = Path(file_path)
        action_file = self.file_path / "action"
        observation_file = self.file_path / "observation"
        reward_file = self.file_path / "rewards"
        terminal_file = self.file_path / "terminal"
        self.actions = np.load(action_file, allow_pickle=True)
        self.observations = np.load(observation_file, allow_pickle=True)
        self.rewards = np.load(reward_file, allow_pickle=True)
        self.terminals = np.load(terminal_file, allow_pickle=True)

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

    def get_max_timestep(self):
        return int(np.max(self.timesteps))

    def get_max_return(self):
        return int(np.max(self.rewards))

    def get_unique_actions(self):
        return np.unique(self.actions)

    def get_data(self):
        return (self.observations,
                self.actions,
                self.returns,
                self.timesteps)


class RLDataset(Dataset):
    def __init__(self, processor, sequence_len, stack_size=4):
        super().__init__()
        self.sequence_len = sequence_len
        self.stack_size = stack_size

        self.observations = torch.tensor(processor.observations,
                                         dtype=torch.float16) / 255
        self.actions = torch.tensor(processor.actions, dtype=torch.int32)
        self.returns = torch.tensor(processor.returns, dtype=torch.float16)
        self.timesteps = torch.tensor(processor.timesteps, dtype=torch.int32)

    def __len__(self):
        return len(self.timesteps) - self.sequence_len - self.stack_size + 1

    def __getitem__(self, idx):
        stacked_obs_seq = torch.stack(
            [
                torch.stack(
                    [self.observations[idx + t + offset] for offset in range(self.stack_size)],
                    dim=0
                )
                for t in range(self.sequence_len)
            ],
            dim=0
        )


        actions_seq = self.actions[idx + self.stack_size - 1: idx + self.stack_size - 1 + self.sequence_len]
        returns_seq = self.returns[idx + self.stack_size - 1: idx + self.stack_size - 1 + self.sequence_len]
        timesteps_seq = self.timesteps[idx + self.stack_size - 1: idx + self.stack_size - 1 + self.sequence_len]

        return (
            stacked_obs_seq,
            actions_seq,
            returns_seq,  #
            timesteps_seq,
        )