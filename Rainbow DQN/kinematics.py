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
from almost import RainbowAgent, ReplayMemory, DuelingDQN


# Visualization and Logging Class
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
                state = state.flatten()  # Flatten the observation
                total_reward = 0
                episode_steps = 0

                for step in range(self.max_steps):
                    action = self.agent.act(state)
                    next_state, reward, done, truncated, _ = self.env.step(action)
                    next_state = next_state.flatten()  # Flatten the next observation

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
        "observation": {"type": "Kinematics", "features": ["x", "y", "vx", "vy"]},
        "action": {"type": "DiscreteMetaAction"},
        "reward_speed_range": [20, 30]
    })

    state_dim = np.prod(env.observation_space.shape)
    action_dim = env.action_space.n

    # Agent initialization
    agent = RainbowAgent(state_dim, action_dim)

    # Training
    trainer = RainbowTrainer(env, agent)
    trainer.train()