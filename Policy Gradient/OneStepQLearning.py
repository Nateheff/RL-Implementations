import torch
import torch.nn as nn
import torch.optim as optim
import numpy
from utils import *
import multiprocessing as mp
import gymnasium as gym
import ale_py

"""
We have global networks whose parameters we will be updating asynchronously
We will have multiple agent each interacting with their own copy of the environment
Each agent will be on a separate thread and will compute a gradient on that thread
We accumulate gradients over many steps to reduce chances of threads overwriting

The agent will:
    Recieve its initial environment
    Choose actions via e-greedy wrt. Q_target
    Continue interacting with its environment for K steps or until episode termination
    One each transition, the agent will receive the next state and a reward
    We use these to calculate the gradient update:
        dG = dG + (d(target - Q_online(s,a))^2 / dG)
    After K steps or termination, we will update the parameters of Q_online using dG

"""

class Q_Function(nn.Module):
    def __init__(self, kernels, kernel_dim, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=32, kernel_size=4, stride=2)
        self.lin = nn.Linear(32*9*9, out_features=256)
        self.out = nn.Linear(in_features=256, out_features=6)
        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.01)
        self.target_counter = 0
        
    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.flatten()
        x = self.lin(x)
        x = self.out(x)
        return x


steps = 5
Q_online = Q_Function(16, 8, 4)
Q_online.share_memory()
Q_target = Q_Function(16, 8, 4)
Q_target.share_memory()

Q_target.load_state_dict(Q_online.state_dict())



def double(next_state, target, online):
    with torch.no_grad():
        action = torch.argmax(online(next_state))
        return target(next_state)[action]

def agent_task(global_q, global_target, counter):
    optimizer = torch.optim.RMSprop(global_q.parameters(), lr=1e-4)

    local_model = Q_Function(16, 8, 4)
    local_model.load_state_dict(global_q.state_dict())
    
    
    step = 0
    gym.register_envs(ale_py)
    env = gym.make("ALE/Pong")
    
    for _ in range(100):

        obs, info = env.reset()
        state = process(obs)
        state = numpy.asanyarray(state, dtype=numpy.float32)
        initial_state = torch.from_numpy(state).unsqueeze(0)
        done = False
        
        while not done:
            
            action, values, greedy = e_greedy(local_model, initial_state)
            new_obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
                break
            next_state = process(new_obs)
            next_state = numpy.asanyarray(next_state, dtype=numpy.float32)
            next_state = torch.from_numpy(next_state).unsqueeze(0)

            target = 0
            if terminated or truncated:
                target = reward
            else:
                target = reward + GAMMA * double(next_state, Q_target, local_model)
            
            loss = (target - values[action])**2
            
            local_model.zero_grad()
            loss.backward()

            step += 1
            if step % steps == 0:
                # optimizer.zero_grad()
                for global_param, local_param in zip(global_q.parameters(),local_model.parameters()):
                    if local_param.grad is not None:
                        #Usual update: global_param.grad += local_param.grad.clone()
                        global_param._grad = local_param.grad #Only in Hogwild as this overwrite global grad in place. 

                optimizer.step()
                local_model.zero_grad()

            with counter.get_lock():
                counter.value += 1
                if counter.value % 500 == 0:
                    global_target.load_state_dict(global_q.state_dict())

            
            local_model.load_state_dict(global_q.state_dict())
            initial_state = next_state



def learn_async():
    try:
        workers = []
        target_counter = mp.Value('i', 0)
        for _ in range(3):
            p = mp.Process(target=agent_task, args=(Q_online, Q_target, target_counter,))
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

