"""

REINFORCE Leave-One-Out (RLOO)

This algorithm returns to the foundations of policy gradient methods and strips away
must of the superfluous complexity introduced to PG methods.

DISCLAIMER: RLOO is actually not well suited for the environment I'm using it for here.
RLOO is best applied in RLHF fine-tuning of already Suprvised Fine-Tuned LLMs, not
randomly initialized policies learning to play atari games. THIS IS FOR PRACTICE AND DEMO ONLY

RLOO takes advantage of the fact that in the RLHF environemt, our initial policy is a relatively
robust SFT LLM that likely won't have a large variance in outputs and cause large gradient steps.
This allows the algorithm to be much simpler and emperically is a valid assumption.

Leave-One-Out refers to the baseline used in the update formula. The formula involves averaging over
many samples to a single state and the baseline that is subtracted from the reward at each sample
is calculated using all other rewards from all other samples EXECPT for the current sample's. (i.e. we
leave that one out)

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *


class Policy(nn.Module):
    def __init__(self, kernels, kernel_dim, stride):
        super().__init__()
        # self.hidden_size = 128
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, out_features=256)
        
        self.down = nn.Linear(in_features=256, out_features=6)
        self.probs = nn.Softmax(dim=-1)

        self.optimizer = optim.Adam(self.parameters())
        
    def forward(self, x:torch.Tensor):

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)        
        out = self.down(x)
        probs = self.probs(out)
        return probs
    


policy = Policy(16, 8, 4)
old_policy = Policy(16, 8, 4)
old_policy.load_state_dict(policy.state_dict())
BETA = 0.05
k = 10


def get_batch_RLOO(k, policy, ref_policy):
    global env
    obs, info = env.reset()
    state = process(obs)
    
    state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)

    probs = policy(state)
    old_probs = ref_policy(state)
    dist = torch.distributions.Categorical(probs)
    dist_old = torch.distributions.Categorical(old_probs)
    actions = []
    rewards = []
    ratios = []
    env_pre = env
    for _ in range(k):
        env = env_pre
        action = dist.sample()
        actions.append(action)
        next_obs, reward, terminated, truncated, info = env.step(action)
        prob = dist.log_prob(action)
        old_prob = dist_old.log_prob(action)
        ratio = prob / old_prob

        reward = reward - BETA * ratio
        rewards.append(reward)
        ratios.append(ratio)

    return torch.tensor(rewards), torch.stack(ratios)


def RLOO(steps):

    for i in range(steps):
        
        #Each reward is not the simple reward, but r - B * log_ratio (A "soft" reward)
        rewards, ratios = get_batch_RLOO(k, policy, old_policy)

        loss = 0
        total = rewards.sum()
        for j in range(len(rewards)):
            reward = rewards[j]

            b = reward - (total - reward) / (k - 1)
            loss = loss + b * ratios[j]
        
        loss = loss / k
        
        policy.optimizer.zero_grad()
        loss.backward()
        policy.optimizer.step()

        


            

if __name__ == "__main__":
    RLOO(1000)