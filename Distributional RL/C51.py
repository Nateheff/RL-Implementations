"""
Distributional RL & C51 (Categoircal Algorithm w/ 51 atoms)

Distributional RL focusses on learning the return distribution, instead of the expected return.
We are finding an approximate value distrubiton using a discrete distribution which is parameterized using
a support.
The support in C51 has 51 atoms with a minimum value of -10 and max of 10

Paraneterized distributional RL is learning to output probabilities of each atom. 
Our output will be of shape ( action space x # of atoms )

Our support is 51 scalars and we learn a probability distribution over each of these atoms, 
so our model will output a row for each action in the action space that contains 51 values 
that are the probability distribution of each action that we will then project onto the support
This projection will then be summed to choose the best action.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *
import math

class ValueDistribution(nn.Module):
    def __init__(self, kernels, kernel_dim, stride, action_space, n_atoms):
        super().__init__()
        # self.hidden_size = 128
        self.action_space = action_space
        self.n_atoms = n_atoms
        self.output_dim = self.action_space * n_atoms
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, out_features=512)
        
        self.down = nn.Linear(in_features=512, out_features=self.output_dim)
        

        self.optimizer = optim.Adam(self.parameters())
        
    def forward(self, x:torch.Tensor):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)        
        out = self.down(x)
        out = out.reshape((self.action_space, self.n_atoms))
        return out
    
V_MIN = -10
V_MAX = 10
N = 51

delta_z = (V_MAX - V_MIN) / (N - 1)

support = torch.arange(V_MIN, V_MAX + delta_z, delta_z)
bellman_support = torch.arange(V_MIN, V_MAX+ delta_z, delta_z)
value_dist = ValueDistribution(16, 8, 4, 6, N)

def fill_buffer(buffer_size, num_episodes, D=None):
    """
    We fill our repaly buffer so that we can sample for traning
    Each transition is (state, action, reward, next_state)
    """

    steps_per_episode = buffer_size // num_episodes

    while len(D) < buffer_size:
        new_obs, info = env.reset()
        new_obs = process(new_obs)
        new_obs = torch.from_numpy(new_obs).unsqueeze(0).unsqueeze(0)
        for _ in range(steps_per_episode):
            probs = value_dist(new_obs)
            final = probs * support
            values = torch.sum(final, dim=-1)


            action = torch.argmax(values, dim=-1)
            next_obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

            next_obs = process(next_obs)
            next_obs = torch.from_numpy(next_obs).unsqueeze(0).unsqueeze(0)

            new_transition = (new_obs, action, reward, next_obs)

            D.append(new_transition)
            new_obs = next_obs


def get_randoms(buffer, batch_size):

    indices = numpy.random.randint(low=0, high=len(buffer), size=batch_size)

    batch = [buffer[i] for i in indices]

    return batch


def parse_batch(batch):
    batch_len = len(batch)
    batch_states = torch.stack([batch[transition][0] for transition in range(batch_len)])
    batch_next = torch.stack([batch[transition][3] for transition in range(batch_len)])

    batch_actions = torch.stack([batch[transition][1] for transition in range(batch_len)])
    batch_rewards = torch.tensor([batch[transition][2] for transition in range(batch_len)], dtype=torch.float32)

    return batch_states, batch_actions, batch_rewards, batch_next


def C51(steps):

    fill_buffer(250, 5, D)

    for i in range(steps):
        new_obs, info = env.reset()
        new_obs = process(new_obs)
        new_obs = torch.from_numpy(new_obs).unsqueeze(0).unsqueeze(0)

        probs = value_dist(new_obs)
        final = probs * support
        values = torch.sum(final, dim=-1)


        action = torch.argmax(values, dim=-1)
        next_obs, reward, terminated, truncated, info = env.step(action)
        next_obs = process(next_obs)
        next_obs = torch.from_numpy(next_obs).unsqueeze(0).unsqueeze(0)
        
        new_transition = (new_obs, action, reward, next_obs)

        D.append(new_transition)

        batch = get_randoms(D, batch_size)

        states, actions, rewards, next_states = parse_batch(batch)
        

        for t in range(len(states)):

            probs = value_dist(next_states[t])
            final = probs * support
            values = torch.sum(final, dim=-1)
            action = torch.argmax(values, dim=-1)
            m = torch.zeros(N)

            for j in range(N):
                bellman_support[j] = rewards[t] + GAMMA * support[j]
                precise_index = (bellman_support[j] - V_MIN)/delta_z
                l = math.floor(precise_index)
                u = math.ceil(precise_index)
                m[t, l] = m[t, l] + probs[j][action] * (u - precise_index)
                m[t, u] = m[t, u] + probs[j][action] * (l - precise_index)

            probs_current = value_dist(states[t])
            p = torch.log(probs_current[actions[t]])
            output = - (m * p).sum()

            
if __name__ == "__main__":
    C51(100)