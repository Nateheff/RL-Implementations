import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
from utils import *
import multiprocessing as mp
import gymnasium as gym
import ale_py
from A3C import LSTM

class Policy(nn.Module):
    def __init__(self, kernels, kernel_dim, stride):
        super().__init__()
        self.hidden_size = 128
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, out_features=256)
        
        self.out = nn.Linear(in_features=self.hidden_size, out_features=6)
        self.probs = nn.Softmax(dim=-1)
        self.adam = optim.Adam(self.parameters())

        
    def forward(self, x:torch.Tensor):

        

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.lin(x)        
        out = self.out(x)
        probs = self.probs(out)
        return out, probs


policy = Policy(16, 8, 4)
delta = 0.05

def fisher_vector_product(v, g_flat, damping=1e-2):


    gv = torch.dot(g_flat, v)
    fvp = torch.autograd.grad(gv, policy.parameters(), retain_graph=True)
    flat_fvp = torch.cat([f.view(-1) for f in fvp])

    regularization = damping * v

    return flat_fvp + regularization

def conjugate_gradient(g, n_steps, residual_min=1e-10):
    """
    Solves F x = g using the Conjugate Gradient method.
    
    Args:
        
        g: right-hand side vector (gradient of surrogate loss)
        nsteps: maximum number of iterations
        residual_tol: tolerance for convergence

    Returns:
        x: approximate solution to Fx = g
    """

    x = torch.zeros_like(g)
    r = g.clone()
    v = r.clone()
    rs_old = torch.dot(r, r)

    for i in range(n_steps):
        fv = fisher_vector_product(v, g)
        alpha = rs_old / torch.dot(v, fv)
        x += alpha * v
        r -= alpha * fv
        rs_new = torch.dot(r, r)

        if rs_new < residual_min:
            break

        v = r + (rs_new / rs_old) * v
        rs_old = rs_new
    return x


def TRPO():
    global policy

    """
    First, we collect our batches
    We need batches of transitions along with their rewards, log_probs, and actions
    We will sample transitions to be later used during model training.
    When we calculate the loss, we will use both the log probability of the choose actions during sampling
    and the log probabilities of the chosen actions during training. (At first run these two will be equal)
    We will also calculate advantages as we sample.
    We need data from both the old (sampling) policy and the new (training) policy
    """

    states, actions, advantages, log_probs_old, rewards = get_batches_TRPO(2048, policy)
    
    q_values, probs = policy(states)
    
    dist = torch.distributions.Categorical(probs)
    log_probs = dist.log_prob(actions)

    #Calculate ratios of new and old action probability distributions and normalize advantages
    ratio = torch.exp(log_probs - log_probs_old)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    #Calculate surrogate loss (negative for gradient ascent and mean due to expectation)
    loss = -(ratio * advantages).mean()

    #compute policy gradient
    g = torch.autograd.grad(loss, policy.parameters())
    #detatch to have g_flat treated as a constant in Pytorch gradient calculation and backprop
    g_flat = torch.cat([grad.view(-1) for grad in g]).detach()

    Fg = conjugate_gradient(g_flat, 10) #step direction

    g_Fg = torch.dot(g, Fg) 

    trust_scale = torch.sqrt(2 * delta / (g_Fg + 1e-8))
    step = trust_scale * Fg

    with torch.no_grad():
        offset = 0
        for param in policy.parameters():
            n_param = param.numel()
            step_segment = step[offset: offset + n_param].view_as(param)
            param += step_segment
            offset += n_param
    
    with torch.no_grad():
        states_detached = states.detch()
    

    q_values_new, _ = policy(states_detached)
    values = q_values_new.gather(1, actions.unsqueeze(1)).squeeze()
    returns = torch.tensor(rewards)

    q_loss = F.mse_loss(values, returns)
    policy.adam.zero_grad()
    q_loss.backward()
    policy.adam.step()


TRPO()