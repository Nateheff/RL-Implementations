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


def PPO(steps):
    global policy

    """
    1. We collect a batch of states, actions, advantages, log probs, and rewards following the old policy

    2. We then use our current policy to get new log probs for these same states and actions

    3. We then calculate our surrogate loss
    NOTE: This loss does not directly optimize rewards, it instead looks at how much the probability of actions
    changed weighted by how good those actions were. We are using gradient ascent so we are trying to maximize this.
    
    4. We then compute our first order derivatives

    5. We then compute out step direction using Conjugate Gradient and Fisher vector product to
    avoid having to calcualte the full fisher information matrix.
    Our update is bounded by the KL divergence of the two policies and this bound is represented in
    the update as step^T * F * step <= 2 * delta
    
    6.After we calcualte our step, we update our parameters 
    
    We use an additional Adam update since our model is also returning Q_values.
    """
    for i in range(steps):
        
        old_policy.load_state_dict(policy.state_dict())
        for param in old_policy.parameters():
            param.requires_grad = False

        states, actions, advantages, log_probs_old, returns = get_batches_GAE(2048, old_policy)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        advantages = advantages.detach()

        value, probs = policy(states)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        #Calculate ratios of new and old action probability distributions and normalize advantages
        ratio = torch.exp(log_probs - log_probs_old.detach())
        clipped = ratio.clip(1-CLIP_EPS, 1+CLIP_EPS)

        trpo_objective = (ratio * advantages)
        clipped_objective = (clipped * advantages)

        #Calculate surrogate loss (negative for gradient ascent and mean due to expectation)
        loss = -torch.min(trpo_objective, clipped_objective).mean()
        print(loss)
        #compute policy gradient
        loss.backward()
        
        policy.optimizer.step()
        policy.optimizer.zero_grad()



PPO(5)