import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy
from utils import *
import multiprocessing as mp
import gymnasium as gym
import ale_py

ALPHA = 0.99
LR = 0.001

class LSTM(nn.Module):
    def __init__(self, hidden_size, input_size):
        super().__init__()
        self.concat_size = hidden_size + input_size
        """
        Using one set of weights is mor efficient and empirically doesn't impact performance significantly.
        The alternative is having a separate set of parameters (in the form of a nn.Linear(self.concat_size, hidden_size) for each gate)
        """
        self.weights = nn.Linear(self.concat_size, 4*hidden_size) # 4x because of chucnking

    def forward(self, x, previous_hidden, previous_cell):
        
        x = torch.cat((x,previous_hidden), dim=-1)

        state = self.weights(x)
        forget, input, output, candidate = torch.chunk(state, chunks=4, dim=-1)

        """
        The candidate will suggest what information the new cell should have (considering information from the previous hidden state and the new input),
        the input gate will decide what and how much of this proposed new state to keep,
        the forget gate will remove the unnecessary parts of the old memory,
        and the output finally decides which parts should be out.
        """
        forget = torch.sigmoid(forget)
        input = torch.sigmoid(input)
        output = torch.sigmoid(output)
        tan = torch.tanh(candidate)

        cell = forget * previous_cell + input * torch.tanh(candidate)
        hidden = output * torch.tanh(cell)
        
        """
        The hidden state is the LSTM layer's current output. It is more relevant to the current input.
        The cell state is the long-term memory that holds the previous information that is used to calculate 
        the hidden state.
        """
        return hidden, cell



class ActorCritic(nn.Module):
    def __init__(self, kernels, kernel_dim, stride):
        super().__init__()
        self.hidden_size = 128
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, out_features=256)
        self.lstm = LSTM(self.hidden_size, input_size=256)
        self.value = nn.Linear(in_features=self.hidden_size, out_features=1)
        self.policy_lin = nn.Linear(in_features=self.hidden_size, out_features=6)
        self.policy = nn.Softmax(dim=-1)
        
        # Initialize RMSprop running averages for each parameter
        self.running_avg = {}
        for name, param in self.named_parameters():
            self.running_avg[name] = torch.zeros_like(param.data)
        

        
    def forward(self, x:torch.Tensor, hidden_state=None, cell_state=None):

        if hidden_state is None:
            hidden_state = torch.zeros(self.hidden_size)
        if cell_state is None:
            cell_state = torch.zeros(self.hidden_size)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.flatten()
        x = self.lin(x)
        new_hidden, new_cell = self.lstm(x, hidden_state, cell_state)
        
        value = self.value(new_hidden)
        policy_down = self.policy_lin(new_hidden)
        policy = self.policy(policy_down)
        return value, policy, new_hidden, new_cell
    
global_model = ActorCritic(16, 8, 4)
global_model.share_memory()

# Share the RMSprop running averages across processes
for name, running_avg in global_model.running_avg.items():
    running_avg.share_memory_()

n = 5

def A3C(global_params):
    optimizer = torch.optim.RMSprop(global_params.parameters(), lr=1e-4)

    local_model = ActorCritic(16, 8, 4)
    local_model.load_state_dict(global_params.state_dict())
    
    for name in local_model.running_avg:
        local_model.running_avg[name] = global_params.running_avg[name]
    

    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong")
    for _ in range(50):


        local_model.load_state_dict(global_params.state_dict())


        hidden_state = torch.zeros(local_model.hidden_size)
        cell_state = torch.zeros(local_model.hidden_size)



        obs, info = env.reset()
        state = process(obs)
        state = numpy.asanyarray(state, dtype=numpy.float32)
        initial_state = torch.from_numpy(state).unsqueeze(0)
        
        done = False
        rewards = []
        states = []
        policies = []
        actions = []
        t = 0
        #We go n steps or until termination collecting states and rewards
        while not done and t < n:
            value, policy, hidden_state, cell_state = local_model(initial_state, hidden_state, cell_state)
            policies.append(policy)
            action = torch.multinomial(policy, num_samples=1)
            new_obs, reward, terminated, truncated, info = env.step(action)
            actions.append(action)
            next_state = process(new_obs)
            next_state = numpy.asanyarray(next_state, dtype=numpy.float32)
            next_state = torch.from_numpy(next_state).unsqueeze(0)
            rewards.append(reward)
            states.append(value)
            

            t += 1
            initial_state = next_state

            if terminated or truncated:
                done = True
                break
        #Once we have our states and rewards from n steps (or however many before terminating), we calculate our final reward
        if done:
            reward = 0 #Reward is 0 if we reached termination
        else:
            value, _, _, _ = local_model(initial_state, hidden_state, cell_state)
            reward = value.item()
            
        rewards.reverse() #We want to accumulate reward and loss backward starting at our final state.
        states.reverse()
        actions.reverse()
        R = reward
        advantage = 0
        loss = 0
        for i in range(len(rewards)):
            R = rewards[i] + GAMMA * R
            advantage = R - states[i].item()

            action_idx = actions[i].item()
            log_policy = torch.log(policies[i].squeeze(0)[action_idx] + 1e-8)
            loss_policy = -log_policy * advantage

            loss_value = F.mse_loss(states[i], torch.tensor([R], dtype=torch.float32))  # shape match

            loss += loss_policy + loss_value

        #Compute new gradients
        local_model.zero_grad()
        loss.backward()
        
        with torch.no_grad():
            for (global_name, global_param), (local_name, local_param) in zip(local_model.named_parameters(), global_model.named_parameters()):
                if local_param.grad is not None:
                    
                    #Calculae running average of gradients for each paramter to use in update
                    g = local_model.running_avg[local_name]

                    g = ALPHA * g + (1 - ALPHA) * local_param.grad.pow(2)
                    std = g.sqrt().add(1e-8)
                    update = -LR * (local_param.grad / std)
                    if global_param.grad is None:
                        global_param.grad = update.clone()
                    else:
                        global_param.grad = global_param.grad + update
                        

        optimizer.step()
        optimizer.zero_grad()
        local_model.zero_grad()

        



def learn_async():
    try:
        workers = []
        
        for _ in range(1):
          
            
            
            p = mp.Process(target=A3C, args=(global_model,))

            p.start()
            workers.append(p)

        for p in workers:
            p.join()

        
    except Exception as e:
        for p in workers:
            p.terminate()
            p.join()
        print(e)
    

learn_async()