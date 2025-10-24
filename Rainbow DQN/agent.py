import gymnasium as gym
import numpy as np
import highway_env
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import torchvision.transforms as transforms

# Convolutional Neural Network for Image-Based Input
class ImageDuelingDQN(nn.Module):
    def __init__(self, action_dim):
        super(ImageDuelingDQN, self).__init__()
        # Convolutional layers for image processing
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size(), 512)
        
        # Advantage and Value streams
        self.advantage_fc = nn.Linear(512, action_dim)
        self.value_fc = nn.Linear(512, 1)

    def feature_size(self):
        # Compute the flattened size of convolutional features
        return self.conv_layers(torch.zeros(1, 1, 84, 84)).view(1, -1).size(1)

    def forward(self, x):
        # Ensure input is in the right format and size
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        
        advantage = self.advantage_fc(x)
        value = self.value_fc(x)
        return value + advantage - advantage.mean()

# Replay Memory (similar to previous implementation)
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
        if len(self.memory) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.position]
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

# Rainbow Agent with Image-Based Input
class RainbowAgent:
    def __init__(self, action_dim, lr=1e-4, gamma=0.99, batch_size=64, memory_size=100000):
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size

        # Networks
        self.policy_net = ImageDuelingDQN(action_dim).to("cuda")
        self.target_net = ImageDuelingDQN(action_dim).to("cuda")

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)

        self.memory = ReplayMemory(memory_size)

        self.update_target()

    def preprocess_image(self, image):
        # Convert to grayscale and tensor
        if len(image.shape) == 3:
            # If color image, convert to grayscale
            image = np.mean(image, axis=2)
        
        # Resize and normalize
        image = np.resize(image, (84, 84))
        image = (image - image.min()) / (image.max() - image.min())
        
        # Convert to tensor
        image_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0).to("cuda")
        return image_tensor

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def act(self, state, epsilon=0.01):
        if random.random() > epsilon:
            state = self.preprocess_image(state)
            with torch.no_grad():
                return self.policy_net(state).argmax(dim=1).item()
        else:
            return random.randrange(self.action_dim)

    def compute_loss(self, samples, indices, weights):
        # Preprocess states and next_states
        states = torch.stack([self.preprocess_image(s) for s, _, _, _, _ in samples]).squeeze(1)
        actions = torch.tensor([a for _, a, _, _, _ in samples]).to("cuda")
        rewards = torch.tensor([r for _, _, r, _, _ in samples], dtype=torch.float32).to("cuda")
        next_states = torch.stack([self.preprocess_image(ns) for _, _, _, ns, _ in samples]).squeeze(1)
        dones = torch.tensor([d for _, _, _, _, d in samples], dtype=torch.float32).to("cuda")
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

        # Update weights and compute TD error
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute priorities
        with torch.no_grad():
            states = torch.stack([self.preprocess_image(s) for s, _, _, _, _ in samples]).squeeze(1)
            actions = torch.tensor([a for _, a, _, _, _ in samples]).to("cuda")
            next_states = torch.stack([self.preprocess_image(ns) for _, _, _, ns, _ in samples]).squeeze(1)
            rewards = torch.tensor([r for _, _, r, _, _ in samples], dtype=torch.float32).to("cuda")
            dones = torch.tensor([d for _, _, _, _, d in samples], dtype=torch.float32).to("cuda")

            current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * self.gamma * next_q
            td_errors = (current_q - target_q).abs().cpu().numpy()

        self.memory.update_priorities(indices, td_errors.tolist())

# Training Loop
class RainbowTrainer:
    def __init__(self, env, agent, num_trails=3, episodes=1_000, max_steps=500):
        self.env = env
        self.agent = agent
        self.num_trails = num_trails
        self.episodes = episodes
        self.max_steps = max_steps
        
        # Logging
        self.trail_rewards = []
        self.episode_rewards = []

        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)

    def train(self):
        for trail in range(self.num_trails):
            trail_episode_rewards = []
            
            # Use tqdm for progress tracking
            trail_progress = tqdm(
                range(self.episodes), 
                desc=f'Trail {trail+1}/{self.num_trails}', 
                total=self.episodes, 
                unit='episode', 
                colour='green'
            )
            
            for episode in trail_progress:
                state, _ = self.env.reset()
                total_reward = 0
                episode_steps = 0

                for step in range(self.max_steps):
                    action = self.agent.act(state)
                    next_state, reward, done, truncated, _ = self.env.step(action)

                    self.agent.memory.push(state, action, reward, next_state, done)
                    self.agent.learn()

                    state = next_state
                    total_reward += reward
                    episode_steps += 1

                    if done or truncated:
                        break

                self.agent.update_target()
                trail_episode_rewards.append(total_reward)

                # Update tqdm progress bar with current episode metrics
                trail_progress.set_postfix({
                    'Reward': f'{total_reward:.2f}', 
                    'Steps': episode_steps
                })

            self.trail_rewards.append(trail_episode_rewards)
            trail_progress.close()

        self.plot_learning_curves()

    def plot_learning_curves(self):
        plt.figure(figsize=(15, 10))
        
        # Average rewards per trail
        plt.subplot(2, 1, 1)
        avg_rewards = np.array(self.trail_rewards)
        plt.plot(np.mean(avg_rewards, axis=0), label='Mean Reward')
        plt.fill_between(
            range(len(avg_rewards[0])), 
            np.mean(avg_rewards, axis=0) - np.std(avg_rewards, axis=0),
            np.mean(avg_rewards, axis=0) + np.std(avg_rewards, axis=0),
            alpha=0.3
        )
        plt.title('Learning Curve - Average Rewards')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')
        plt.legend()

        # Box plot of trail rewards
        plt.subplot(2, 1, 2)
        plt.boxplot(self.trail_rewards)
        plt.title('Reward Distribution Across Trails')
        plt.xlabel('Episodes')
        plt.ylabel('Reward')

        plt.tight_layout()
        plt.savefig('results/rainbow_learning_curves.png')
        plt.close()

        # Save trail rewards for further analysis
        np.save('results/trail_rewards.npy', self.trail_rewards)

# Main execution
if __name__ == "__main__":
    # Environment setup
    env = gym.make("highway-v0", config = {
        "observation": {"type": "GrayscaleObservation",
                        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
                        "stack_size": 4,
                        "observation_shape": (84, 84)},  # Use grayscale image observations
        "action": {"type": "DiscreteMetaAction"},
        "reward_speed_range": [20, 30]
    })

    action_dim = env.action_space.n

    # Agent initialization
    agent = RainbowAgent(action_dim)

    # Training
    trainer = RainbowTrainer(env, agent)
    trainer.train()