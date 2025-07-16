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

Action space is 17 elements ranging -0.4 to 0.4
"""

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
from utils import *

LOG_STD_MIN = -20
LOG_STD_MAX = 5

log_alpha = torch.tensor([0.01], requires_grad=True) #In log space for numerical stability
alpha_optim = optim.Adam([log_alpha], lr=1e-4)
target_entropy = -float(17)  # Typically: -|A|

class Policy(nn.Module):
    def __init__(self, hidden_size, action_size):
        super().__init__()

        self.relu = nn.ReLU()
        self.lin1 = nn.Linear(in_features=348, out_features=hidden_size)
        self.lin2 = nn.Linear(hidden_size, hidden_size)
        self.deviation = nn.Linear(hidden_size, action_size)
        self.mean = nn.Linear(hidden_size, action_size)
        self.optim = optim.Adam(self.parameters())

    def forward(self, x):

        x = self.lin1(x)
        x_1 = self.relu(x)
        x = self.lin2(x)
        x = self.relu(x)
        x = x + x_1
        mean = self.mean(x)
        log_deviation = self.deviation(x)
        log_deviation = torch.clamp(log_deviation, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_deviation)
        
        
        action, pre_tanh = self.get_action(mean, std)
        log_prob = self.get_log_probs(mean, std, 1e-6, pre_tanh, action)

        return action, log_prob
    
    def get_action(self, mean, deviation):
        
        pre_action = torch.normal(mean=mean, std=deviation)
        action = torch.tanh(pre_action) * 0.4
        return action, pre_action
    
    def get_log_probs(self, mean, std, eps, pre_tanh, action):
        normal = torch.distributions.Normal(mean, std)

        log_prob = normal.log_prob(pre_tanh)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        # correction for Tanh squashing
        log_prob -= torch.sum(
            torch.log(1 - action.pow(2) + 1e-6),
            dim=-1, keepdim=True
        )

        return log_prob

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


policy = Policy(512, 17)
Q_1 = Critic(512, 17, 348)
Q_2 = Critic(512, 17, 348)

value_function = StateValue(512)
value_function_target = StateValue(512)

env = gym.make("Humanoid-v5")

observation, info = env.reset(seed=42)

obs = torch.from_numpy(numpy.stack(observation, dtype=numpy.float32))



D = deque(maxlen=250)

def fill_buffer(buffer_size, num_episodes):
    global buffer
    steps_per_episode = buffer_size // num_episodes

    while len(D) < buffer_size:
        new_obs, info = env.reset()
        new_obs = torch.from_numpy(numpy.stack(new_obs, dtype=numpy.float32))
        
        for _ in range(steps_per_episode):
            action, _ = policy(new_obs)

            action_step = action.detach().numpy()
            next_obs, reward, terminated, truncated, info = env.step(action_step)

            if terminated or truncated:
                break

            next_obs = torch.from_numpy(numpy.stack(next_obs, dtype=numpy.float32))

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
    



def SAC(steps):
    
    n_randoms = 10
    fill_buffer(250, 10)
    for i in range(steps):

        new_obs, info = env.reset()
        new_obs = torch.from_numpy(numpy.stack(new_obs, dtype=numpy.float32))

        action, log_prob = policy(new_obs)

        action_step = action.detach().numpy()
        next_obs, reward, terminated, truncated, info = env.step(action_step)
        next_obs = torch.from_numpy(numpy.stack(next_obs))

        new_transition = (new_obs, action, reward, next_obs)

        D.append(new_transition)

        batch = get_randoms(D, n_randoms)
        batch_states, batch_actions, batch_rewards, batch_next = parse_batch(batch)

        # for i in range(n_randoms):
        new_actions, new_log_probs = policy(batch_states)

        batch_pairs = torch.cat((batch_states, new_actions), dim=1)

        values = value_function(batch_states).squeeze()

        q_values1, q_values2 = Q_1(batch_pairs).squeeze(), Q_2(batch_pairs).squeeze()

        
        log_probs = new_log_probs.squeeze()
        q_values = torch.minimum(q_values1, q_values2)

        value_part = (q_values - log_probs)
        value_obj = F.mse_loss(values, value_part)

        with torch.no_grad():
            values_target = value_function_target(batch_next).squeeze()
            q_part = (batch_rewards + GAMMA * values_target)
        q_obj_1 = F.mse_loss(q_values1, q_part)
        q_obj_2 = F.mse_loss(q_values2, q_part)
        
        alpha = torch.exp(log_alpha)
        policy_obj = (q_values - alpha.detach() * log_probs).mean()

        policy.optim.zero_grad()
        value_function.optim.zero_grad()
        Q_1.optim.zero_grad()
        Q_2.optim.zero_grad()
        
        total_loss = value_obj + q_obj_1 + q_obj_2 + policy_obj
        total_loss.backward()

        alpha_loss = -(log_alpha * (log_probs + target_entropy).detach()).mean()
        alpha_optim.zero_grad()
        alpha_loss.backward()
        alpha_optim.step()


        policy.optim.step()
        value_function.optim.step()
        Q_1.optim.step()
        Q_2.optim.step()


        tao = 0.001
        with torch.no_grad():
            for parameter, parameter_t in zip(value_function.parameters(),value_function_target.parameters()):
                parameter_t.data.copy_(tao * parameter.data + (1 - tao) * parameter_t.data)



SAC(100)