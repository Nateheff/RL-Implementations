
import torch
import torch.nn as nn
import torch.optim as optim
import numpy.random as rand
import pickle

from utils import *


class Q_Function(nn.Module):
    def __init__(self, kernels, kernel_dim, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=6)
        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4)
        self.target_counter = 0
        
    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        if len(x.shape) > 3:
            x = x.view(x.shape[0], -1)
        else:
            x = x.flatten()
        x = self.lin(x)
        x = self.out(x)
        return x
    


        

q_current = Q_Function(16, 8, 4)
q_target = Q_Function(16, 8, 4)


def learn_DQN(q_target:Q_Function, q_current:Q_Function, transitions=None):

    q_current.optimizer.zero_grad()
    rewards = torch.tensor([transition[2] for transition in transitions])
    next_obs = torch.stack([transition[3] for transition in transitions])
    current_obs = torch.stack([transition[0] for transition in transitions])
    with torch.no_grad():
        targets = torch.max(q_target(next_obs), dim=-1).values
    currents = q_current(current_obs)
    
    target = rewards + GAMMA * targets
    current = torch.stack([current[transition[1]] for current, transition in zip(currents, transitions)])

    loss = q_current.loss(current, target)
    
    loss.backward()
    q_current.optimizer.step()
    q_current.target_counter += 1

def learn_DDQN(q_target:Q_Function, q_current:Q_Function, transitions=None):
    """
    DDQN (Double DQN):

    The only thing that changes is our target. Instead of choosing the maximum action with respect to the parameters 
    of our target netowrk and then also using the same parameters to calculate the q-value of this action,
    we use our current network to select the action and the target netwrok to calculate the q-value of said action.
    If we were to use the target netwrok's paramteres for both, we get a large estimate for the value of (S', A') and
    only subtracting only the observed (s,a) pair with respect to the current network's paramters. This overestimation
    causes a bias that is resolved by letting our current network choose the action and our target network evaluate said action.
    """
    q_current.optimizer.zero_grad()

    rewards = torch.tensor([transition[2] for transition in transitions])
    next_obs = torch.stack([transition[3] for transition in transitions])
    current_obs = torch.stack([transition[0] for transition in transitions])
    with torch.no_grad():
        targets = q_target(next_obs)
    currents = q_current(current_obs)
    target_idx = torch.argmax(q_current(next_obs), dim=-1)
    targets = targets[torch.arange(0,len(targets)), target_idx]
    
    target = rewards + GAMMA * targets
    current = torch.stack([current[transition[1]] for current, transition in zip(currents, transitions)])

    loss = q_current.loss(current, target)

    loss.backward()
    q_current.optimizer.step()
    q_current.target_counter += 1


def train(episodes):

    for _ in range (episodes):

        """
        Our input is not a singel frame, but a sequence of k frames. (k = 4 usually)
        """
        input = collect_experience(env)
        action, values, greedy = e_greedy(q_current, input)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            continue
        
        new_transition = get_next(observation, action, reward, input)
        D.append(new_transition)
        if len(D) < 5:
            continue
        randoms = get_randoms(D)
        randoms.append(new_transition)
        learn_DDQN(q_target, q_current, randoms)
        if q_current.target_counter == 100:
            
            q_target.load_state_dict(q_current.state_dict())
            q_current.target_counter = 0
    

def play(n_gamnes):

    for _ in range(n_gamnes):

        obs = env.reset()
        score = 0
        gg = False
        input = collect_experience(env)
        while not gg:
            
            q_vals = q_current(input)
            action = torch.argmax(q_vals)
            new_obs, reward, terminated, truncated, info = env.step(action)

            score += reward
            obs = new_obs
            new = get_next(obs, action, reward, input)
            input = new[3]



if __name__ == "__main__":
    train(500)
    
    