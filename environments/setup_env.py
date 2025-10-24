import gymnasium as gym
import highway_env

def create_highway_env(render_mode='rgb_array'):
    config = {
        "observation": {"type": "Kinematics", "features": ["x", "y", "vx", "vy"]},
        "action": {"type": "DiscreteMetaAction"},
        "reward_speed_range": [20, 30]
    }
    env = gym.make("highway-v0", config=config, render_mode=render_mode)
    return env
