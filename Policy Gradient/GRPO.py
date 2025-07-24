"""
NOTE: GRPO is best suited for an RLHF space with LLMs and NLP. The formulation heavily relies on the use
of a reward function and assumes we are receiving outputs to prompts, not actions for games. This
implmenetation is for an Atari game space and is only for demonstration purposes.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import *

CLIP_EPS = 0.2

class Policy(nn.Module):
    def __init__(self, kernels, kernel_dim, stride):
        super().__init__()
        # self.hidden_size = 128
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, out_features=256)
        
        self.value = nn.Linear(in_features=256, out_features=1)
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
        values = self.value(x)
        return values, probs
    


policy = Policy(16, 8, 4)
old_policy = Policy(16, 8, 4)
delta = 0.05
G = 10

def GRPO(steps):
    global policy

    """
    GRPO differs from PPO in using multiple samples per state and using the normalized reward for each state
    as the advantage.
    For each state in our batch, we take G samples using our old policy. 
    We then calulate the reward for each sample to be (reward - mean_reward_of_state) / states_rewards_std
    So each state will have G actions, advantages, and sets of log probs.
    In our objective calculation, we will first take the mean of the PPO objective for each sample of each state, then
    take the mean of these means to get our final loss. (We first sum over the PPO objective at each state one sample 
    at a time, then we sum over these sum having one sum per sample number (G sums) and get our final total loss)
    
    """
    for i in range(steps):
        
        old_policy.load_state_dict(policy.state_dict())
        for param in old_policy.parameters():
            param.requires_grad = False

        states, actions, advantages, log_probs_old = get_batches_GRPO(2048, old_policy)

        advantages = advantages.detach()

        value, probs = policy(states)
        probs = probs.unsqueeze(1)
        probs = probs.repeat(1, G, 1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        #Calculate ratios of new and old action probability distributions
        ratio = torch.exp(log_probs - log_probs_old.detach())
        
        clipped = ratio.clip(1-CLIP_EPS, 1+CLIP_EPS)

        trpo_objective = (ratio * advantages)
        clipped_objective = (clipped * advantages)

        #Calculate surrogate loss (negative for gradient ascent and mean due to expectation)
        min_samples = torch.min(trpo_objective, clipped_objective)
        loss= -min_samples.mean()
        
        #compute policy gradient
        loss.backward()
        
        policy.optimizer.step()
        policy.optimizer.zero_grad()



GRPO(10)