"""
SOFT ACTOR-CRITIC (SAC)

The authors were focussed on two persitent problems in the space of RL:
1. Sample inefficiency
2. Brittle Convergence Guarantees

To address these issues, they use off-policy learning with replay exerpience to imporve sample efficiency
However, policy gradient methods don't naturally accomodate off-policy learning very well
since policy gradients assume we're updating the same disribution we're sampling from
and this is not the case in off-policy learning, where we are learning from trajectories generated
by previous versions of our model (ie. not our current distribution).
To accomodate off-policy learning in PG, they turn to the Max Entropy Framework.
Max Entropy Framework:
Augment standard max reward-based RL with an entropy term so that the agent learns to maximize
reward while acting with as much entropy (as randomly) as possible.

"""

"""
IMPLEMENTATION NOTES:

New: Need to get a continuous action space
We could have a function fill our replay buffer with a couple different episodes first.

1. Need Experience Replay mechanism
2. Need function approximator (non-linear, q-value approximator)
3. Need policy model

Action space is 17 elements raning -0.4 to 0.4
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy
from utils import *

class Policy(nn.Module):
    def __init__(self, hidden_size, action_size):
        super().__init__()

        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features=348, out_features=hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.deviation = nn.Linear(hidden_size, action_size)
        self.mean = nn.Linear(hidden_size, action_size)

    def forward(self, x):

        x = self.lin1(x)
        x_1 = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x += x_1
        mean = self.mean(x)
        log_deviation = self.deviation(x)
        std = torch.exp(log_deviation)
        return mean, std
    
    def get_action(self, mean, deviation):
        
        action = torch.normal(mean=mean, std=deviation)
        action = torch.tanh(action) * 0.4
        return action

#NOTE: In continuous actions spaces, we input our state AND action to our state-action value function and it returns
# a scalar value estimation of that state-action pair.

class Critic(nn.Module):
    def __init__(self, hidden_size, action_size, state_size):
        super().__init__()

        self.in_size = action_size + state_size 
        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features=self.in_size, out_features=hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = self.lin1(x)
        x_1 = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x += x_1
        x = self.out(x)

        return x


class StateValue(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features=348, out_features=hidden_size)
        self.out = nn.Linear(hidden_size, 1)

    def forward(self, x):

        x = self.lin1(x)
        x = self.relu(x)
        out = self.out(x)

        return out


policy = Policy(512, 17)
Q_1 = StateValue(512)
Q_2 = StateValue(512)

critic = Critic(512, 17, 348)
critic_target = Critic(512, 17, 348)

env = gym.make("Humanoid-v5")

observation, info = env.reset(seed=42)

obs = torch.from_numpy(numpy.stack(observation, dtype=numpy.float32))



buffer = []

def fill_buffer(buffer_size, num_episodes):
    global buffer
    steps_per_episode = buffer_size // num_episodes

    while len(buffer) < buffer_size:
        new_obs, info = env.reset()
        new_obs = torch.from_numpy(numpy.stack(new_obs, dtype=numpy.float32))
        
        for _ in range(steps_per_episode):
            action = policy(new_obs)

            next_obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

            next_obs = torch.from_numpy(numpy.stack(next_obs))

            new_transition = (new_obs, action, reward, next_obs)

            D.append(new_transition)
            new_obs = next_obs


def get_randoms(buffer, batch_size):

    indices = numpy.random.randint(low=0, high=len(buffer), size=batch_size)

    batch = [buffer[i] for i in indices]

    return batch


def parse_batch(batch):
    batch_states = torch.stack([batch[transition][0] for transition in batch])
    batch_next = torch.stack([batch[transition][3] for transition in batch])

    batch_actions = torch.tensor([batch[transition][1] for transition in batch])
    batch_rewards = torch.tensor([batch[transition][2] for transition in batch])

    return batch_states, batch_actions, batch_rewards, batch_next
    

def SAC(steps):
    
    n_randoms = 10

    for i in range(steps):

        new_obs, info = env.reset()
        new_obs = torch.from_numpy(numpy.stack(new_obs))

        action = policy(new_obs)

        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = torch.from_numpy(numpy.stack(next_obs))

        new_transition = (new_obs, action, reward, next_obs)

        D.append(new_transition)

        batch = get_randoms(buffer, n_randoms)
        batch_states, batch_actions, batch_rewards, batch_next = parse_batch(batch)

        for i in range(n_randoms):
            pass



