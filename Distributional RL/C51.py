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
        self.softmax = nn.Softmax(dim=1)

        self.optimizer = optim.Adam(self.parameters())
        
    def forward(self, x:torch.Tensor):
        """
        We start with a usual convolutional backbone to represent the atari game screne to our
        output head.
        Our output head states the output of the convolutional backbone, flattens it and
        then learns to output a probability distribution over each action that will
        be projected onto the support. The output is of shape [# of actions x # of atoms]
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)        
        out = self.down(x)
        out = out.reshape((self.action_space, self.n_atoms))
        #Each ROW should sum to one so that each component of our output matrix is the probability mass
        #for that atom for that action. Each column represents the realtive prob. mass of each atom and the row
        #is the total prob mass of all atoms for each action. We are learning to output a prob. dist. for
        #each action that will be projected onto our support.
        out = self.softmax(out)
        return out
    
V_MIN = -10
V_MAX = 10
N = 51

delta_z = (V_MAX - V_MIN) / (N - 1)

#Our fixed support which our model will learn to output a probability distribtution over for each action
support = torch.arange(V_MIN, V_MAX + delta_z, delta_z)
#Our bellman support which will be projected back on to our fixd support
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

            """
            We project our modwel output onto our support and sum over each row (set of atoms). This gives us
            a vector of one scalar for each action that represent the predicted value of that action.
            We then use a greedy action selection.
            """
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
        output = 0

        for t in range(len(states)):
            """
            Since we're working with the distributional bellamn equation which fouses on current reward
            and value distribution of the next state, we first calculate the value distribution of each of the 
            next states in our batch
            """
            probs = value_dist(next_states[t])
            final = probs * support
            values = torch.sum(final, dim=-1)
            action = torch.argmax(values, dim=-1)
            m = torch.zeros(N)

            for j in range(N):
                """
                After we apply out bellam update and get our bellman support, our new support 
                (which we are calling bellman_support) is disjoint with our fixed support. Thus
                we have to project our bellman support onto our fixed support.
                """
                bellman_support[j] = rewards[t] + GAMMA * support[j]
                
                #This is where the bellman support value would lie on our fixed support if it were continuous
                precise_index = (bellman_support[j] - V_MIN)/delta_z
                #Next-lowest neighbor index
                l = math.floor(precise_index)
                #Next neighbor index
                u = math.ceil(precise_index)
                #The action probability 
                p_j = probs[action, j]
                #Distribute the probability mass to nearby neighbors of bellman index
                m[l] = m[l] + p_j * (u - precise_index)
                m[u] = m[u] + p_j * (precise_index - l)
            """
            In our batch learning above, we have gained a probability mass distribution, m, using distributional
            Bellman update which we will now use to calculate our cross entropy loss by multiplying our prob.
            mass distribution by the action probabiltie at our current state
            """
            probs_current = value_dist(states[t])
            p = torch.log(probs_current[actions[t]])
            output += - (m * p).sum()
        print(output)
        value_dist.optimizer.zero_grad()
        output.backward()
        value_dist.optimizer.step()
            
if __name__ == "__main__":
    C51(100)