import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.multiprocessing as mp
import gymnasium as gym
import highway_env
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from datetime import datetime

current_datetime = datetime.now()
formatted_datetime = current_datetime.strftime("%Y%m%d_%H%M%S")

EPISODES = 100
T_MAX = 5
NUM_THREADS = 8
lr = 1e-4
env_id = 'highway-v0' # Replace 'highway' with 'merge' for running merge environemnt
GAMMA = 0.99
STEPS_DATA = []
SEED = 1234

np.random.seed(SEED)

class ActorCritic(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(ActorCritic, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        
        self.fc = nn.Sequential(
            nn.Linear(self._get_conv_output(input_shape), 512),
            nn.ReLU()
        )
        
        self.actor = nn.Linear(512, n_actions)
        self.critic = nn.Linear(512, 1)

        self.states = []
        self.actions = []
        self.rewards = []
    
    def _get_conv_output(self, shape):
        o = self.conv_layers(torch.zeros(1, *shape))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        conv_out = self.conv_layers(x)
        conv_out = conv_out.view(conv_out.size(0), -1)
        fc_out = self.fc(conv_out)
        policy = F.softmax(self.actor(fc_out), dim=-1)
        value = self.critic(fc_out)
        return policy, value

    def save(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def clear(self):
        self.states = []
        self.actions = []
        self.rewards = []
    
    def update(self, done):
        # Calculate reward
        _, v = self.forward(torch.from_numpy(np.vstack(self.states)))
                    
        R = v[-1]*(1-int(done))

        buffered_reward = []
        for reward in self.rewards[::-1]:
            R = reward + GAMMA*R
            buffered_reward.append(R)

        buffered_reward.reverse()
        rewards = torch.tensor(buffered_reward, dtype=torch.float)

        # Calculate loss
        policy, values = self.forward(torch.from_numpy(np.vstack(self.states)))
        values = values.squeeze()

        td = rewards - values
        c_loss = td.pow(2)

        probs = F.softmax(policy, dim=1)

        a = torch.tensor(np.array(self.actions))
        m = torch.distributions.Categorical(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (c_loss + a_loss).mean()

        return rewards, total_loss

    def choose_action(self, state, n_actions):
        policy, value = self.forward(state)
                
        action_probs = policy.detach().numpy()[0]
        action = np.random.choice(n_actions, p=action_probs)

        return action

class SharedRMSProp(optim.RMSprop):
    def __init__(self, params, lr=1e-4, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
        super(SharedRMSProp, self).__init__(params, lr, alpha, eps, weight_decay, momentum, centered)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = torch.tensor(0.0).share_memory_()
                state['square_avg'] = torch.zeros_like(p.data).share_memory_()
                if momentum > 0:
                    state['momentum_buffer'] = torch.zeros_like(p.data).share_memory_()
                if centered:
                    state['grad_avg'] = torch.zeros_like(p.data).share_memory_()


class Agent(mp.Process):
    def __init__(self, global_actor_critic, global_optim, input_shape, 
                 n_actions, gamma, lr, name, global_ep_idx, res_queue, env_id):
        super(Agent, self).__init__()
        
        # Initialization parameters
        self.global_actor_critic = global_actor_critic
        self.global_optim = global_optim
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.gamma = gamma
        self.lr = lr
        self.name = name
        self.global_ep_idx = global_ep_idx
        self.res_queue = res_queue
        self.env_id = env_id
        
        self.local_actor_critic = ActorCritic(input_shape, n_actions)
        
    def run(self):
        env = gym.make(self.env_id, config={"observation": {
        "type": "GrayscaleObservation",
        "observation_shape": (84, 84),
        "stack_size": 4,
        "weights": [0.2989, 0.5870, 0.1140],  # weights for RGB conversion
        "scaling": 1.75}, 
                            "action": {"type": "DiscreteMetaAction"}})

        while self.global_ep_idx.value < EPISODES:
            state, _ = env.reset(seed=SEED)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            done = False
            terminated = False
            ep_reward = 0
            t_step = 1
            
            while (not (done)):
                action = self.local_actor_critic.choose_action(state_tensor, self.n_actions)

                next_state, reward, done, _, _ = env.step(action)

                STEPS_DATA.append([state, action, reward, done])

                next_state = torch.FloatTensor(next_state).unsqueeze(0)
                
                self.local_actor_critic.save(state_tensor, action, reward)

                ep_reward += reward
                t_step += 1

                if t_step % T_MAX == 0 or done:
                    reward, loss = self.local_actor_critic.update(done)                    

                    self.global_optim.zero_grad()
                    loss.backward()

                    for local_param, global_param in zip(self.local_actor_critic.parameters(),self.global_actor_critic.parameters()):
                        global_param._grad = local_param.grad
                    
                    self.global_optim.step()

                    self.local_actor_critic.load_state_dict(self.global_actor_critic.state_dict())
                    
                    self.local_actor_critic.clear()

                state = next_state
            
            self.res_queue.put(ep_reward)

            with self.global_ep_idx.get_lock():
                self.global_ep_idx.value += 1
            print(f"{self.name} - Episode: {self.global_ep_idx.value}\tReward: {ep_reward}")
            
        self.res_queue.put(None)

def moving_average(data, *, window_size = 50):
    """Smooths 1-D data array using a moving average.

    Args:
        data: 1-D numpy.array
        window_size: Size of the smoothing window

    Returns:
        smooth_data: A 1-d numpy.array with the same size as data
    """
    assert data.ndim == 1
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]
 
def main():
    input_shape = (4, 84, 84)
    n_actions = 5
    
    global_actor_critic = ActorCritic(input_shape, n_actions)
    global_actor_critic.share_memory()
    
    optim = SharedRMSProp(global_actor_critic.parameters(), lr=lr, alpha=0.99, eps=0.4)
    
    global_ep = mp.Value('i', 0)
    res_queue = mp.Queue()
    
    workers = [Agent(global_actor_critic,
                     optim,
                     input_shape,
                     n_actions,
                     gamma=GAMMA,
                     lr=lr,
                     name='Process: %02i' % i,
                     global_ep_idx=global_ep,
                     res_queue =res_queue,
                     env_id=env_id) for i in range(NUM_THREADS)]
    
    [w.start() for w in workers]
    
    rewards = []
    
    while True:
        r = res_queue.get()
    
        if r is not None:
            rewards.append(r)
        else:
            break
            
    np.save(env_id + "result_"+str(EPISODES) + "_" +str(NUM_THREADS) + "_" + formatted_datetime, np.array(rewards))

    rewards = moving_average(np.array(rewards))

    plt.plot(rewards)
    plt.ylabel('Result')
    plt.xlabel('Episode')
    plt.show()    

    np.save(env_id + "_steps_"+str(EPISODES) + "_" +str(NUM_THREADS) + "_" + formatted_datetime, np.array(STEPS_DATA))
    
    [w.join() for w in workers]
        
if __name__ == '__main__':
    mp.set_start_method('spawn')
    main()