"""

Twin Delayed Deep Deterministic

Twin: Twin refers to training two indepedent state-action value functions. We train the two 
    independently and use the minimum estimate of the two in the update of the state-value
    function parameters. This is done to mitigate the overestimation bias.

Delayed: Delayed refers to delaying policy updates, allowing value functions estimates to
    converge before the policy is improved. This stabilizes loss and avoids overfitting to 
    a noisy and inaccurate value function.

Deep Deterministic: Action outputs of our policy are deterministic, no stochasticity. Thus, we 
    are using deterministic policy gradient where we take the expected gradient of the state-action
    value function wrt. the action time the gradient of the policy wrt. its parameters.



"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
from utils import *
from SAC import fill_buffer, parse_batch

sigma = 1.5
tao = 0.001

class Policy(nn.Module):
    def __init__(self, hidden_size, action_size):
        super().__init__()

        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features=348, out_features=hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, action_size)
        self.optim = optim.Adam(self.parameters())

    def forward(self, x):
        """
        In TD3, we are using a deterministic policy, so we simply output an action prediction
        from our model. No stochastic sampling.
        """
        x = self.lin1(x)
        x_1 = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = x + x_1
        action = self.out(x)
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
        self.optim = optim.Adam(self.parameters())

    def forward(self, x):

        x = self.lin1(x)
        x_1 = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = x + x_1
        x = self.out(x)

        return x


class StateValue(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features=348, out_features=hidden_size)
        self.out = nn.Linear(hidden_size, 1)
        self.optim = optim.Adam(self.parameters())

    def forward(self, x):

        x = self.lin1(x)
        x = self.relu(x)
        out = self.out(x)

        return out


critic_1 = Critic(512, 17, 348)
critic_2 = Critic(512, 17, 348)

actor = Policy(512, 17)
actor_target = Policy(512, 17)
actor_target.load_state_dict(actor.state_dict())

critic_target_1 = Critic(512, 17, 348)
critic_target_2 = Critic(512, 17, 348)

critic_target_1.load_state_dict(critic_1.state_dict())
critic_target_2.load_state_dict(critic_2.state_dict())

D = deque(maxlen=250)

env = gym.make("Humanoid-v5")

observation, info = env.reset(seed=42)

obs = torch.from_numpy(numpy.stack(observation, dtype=numpy.float32))

def TD3(steps):
    n_randoms = 10
    fill_buffer(250, 10, D)
    for i in range(steps):
        
        
        new_obs, info = env.reset()
        new_obs = torch.from_numpy(numpy.stack(new_obs, dtype=numpy.float32))

        action = actor(new_obs)
        noise = torch.normal(0, sigma, size=action.shape)
        action = action + noise
        action = torch.clamp(action, -0.4, 0.4)

        action_step = action.detach().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action_step)
        next_obs = torch.from_numpy(numpy.stack(next_obs, dtype=numpy.float32))

        new_transition = (new_obs, action, reward, next_obs)

        D.append(new_transition)

        batch = get_randoms(D, n_randoms)
        batch_states, batch_actions, batch_rewards, batch_next = parse_batch(batch)

        """
        We add noise to encourage exploration.
        """
        batch_noise = torch.normal(0, sigma, size=batch_next.shape)
        next_actions = actor_target(batch_next) + batch
        #Our enviroment uses actions between [-0.4, 0.4]
        next_actions = torch.clamp(next_actions, -0.4, 0.4)

        next_pairs = torch.cat(batch_next, next_actions, dim=1)

        q_target_1 = critic_target_1(next_pairs)
        q_target_2 = critic_target_2(next_pairs)

        #Minimum to mitigate overestimation bias
        q_targets = torch.minimum(q_target_1, q_target_2)

        #In the paper, they have a state-value function used in the targets, but
        # in practice, we appx. this function using our state-action functions.
        q_targets = batch_rewards + GAMMA * q_targets

        batch_pairs = torch.cat(batch_states, batch_actions, dim=1)
        q_current_1 = critic_1(batch_pairs)
        q_current_2 = critic_2(batch_pairs)
        
        q_currents = torch.minimum(q_current_1, q_current_2)

        #Soft update using essentially TD error
        q_loss = F.mse_loss(q_targets, q_currents)

        q_loss.backward()
        critic_1.optim.zero_grad()
        critic_2.optim.zero_grad()
        critic_1.step()
        critic_2.step()

        #Delay
        if i % 3 == 0:
            #dQ/da * da/dθ automatically and we want to maximize q, so we minimize the negative.
            policy_loss = -q_current_1.mean()
            actor.optim.zero_grad()
            policy_loss.backward()
            actor.optim.step()

            with torch.no_grad():
                for t_param_1, t_param_2, param_1, param_2 in zip(critic_target_1.parameters(), critic_target_2.parameters(), critic_1.parameters(), critic_2.parameters()):
                    t_param_1.data.copy_(tao * param_1 + (1-tao) * t_param_1)
                    t_param_2.data.copy_(tao * param_2 + (1-tao) * t_param_2)
                
                for target_param, param in zip(actor_target.parameters(), actor.parameters()):
                    target_param.copy_(tao * param + (1 - tao) * target_param)
            
                    

            