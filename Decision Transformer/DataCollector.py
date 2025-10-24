import gymnasium as gym
import highway_env
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
import os

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


output_dir = 'highway_ppo_data'
os.makedirs(output_dir, exist_ok=True)


max_timesteps = 2_400_000
observations = np.zeros((max_timesteps,) + env.observation_space.shape, dtype=np.float32)
actions = np.zeros((max_timesteps, env.action_space.shape[0]), dtype=env.action_space.dtype)
rewards = np.zeros(max_timesteps, dtype=np.float32)
terminated = np.zeros(max_timesteps, dtype=bool)


model = PPO('CnnPolicy', env, verbose=1)

obs, _ = env.reset()
timestep = 0

try:
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
            print(f"Collected {timestep} timesteps")

            np.save(os.path.join(output_dir, f'observations_{timestep}.npy'), observations[:timestep])
            np.save(os.path.join(output_dir, f'actions_{timestep}.npy'), actions[:timestep])
            np.save(os.path.join(output_dir, f'rewards_{timestep}.npy'), rewards[:timestep])
            np.save(os.path.join(output_dir, f'terminated_{timestep}.npy'), terminated[:timestep])

except Exception as e:
    print(f"Error occurred: {e}")

finally:

    print("Saving final collected data...")
    np.save(os.path.join(output_dir, 'observations.npy'), observations[:timestep])
    np.save(os.path.join(output_dir, 'actions.npy'), actions[:timestep])
    np.save(os.path.join(output_dir, 'rewards.npy'), rewards[:timestep])
    np.save(os.path.join(output_dir, 'terminated.npy'), terminated[:timestep])

    env.close()
