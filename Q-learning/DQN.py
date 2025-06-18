import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy.random as rand
from PIL import Image
import numpy
from collections import deque

HEIGHT = 160
WIDTH = 210
DOWN = 110
CROP = 84
CROP_LEFT = (DOWN - CROP) / 2

GAMMA = 0.9
EPSILON = 0.1

gym.register_envs(ale_py)
env = gym.make("ALE/Pong")
observation, info = env.reset(seed=42)

D = deque(maxlen=50)

class Q_Function(nn.Module):
    def __init__(self, kernels, kernel_dim, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
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
    


        

q_current = Q_Function(16, 8, 4)
q_target = Q_Function(16, 8, 4)


def process(observation):

    img = Image.fromarray(observation)
    grayscale = img.convert('L')
    down = grayscale.resize((DOWN,CROP), resample=Image.BILINEAR)
    cropped = down.crop((CROP_LEFT,0, CROP_LEFT + CROP, CROP))
    return cropped


def collect_experience():

    history = []
    observation, info = env.reset(seed=42)

    for _ in range(4):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        history.append(process(observation))

    n_hist = numpy.asanyarray(history, dtype=numpy.float32)
    input = torch.from_numpy(n_hist) #input is an 84 x 84 x 4 tensor
    return input


def e_greedy(state):
    values = q_current(state)
    if rand.binomial(1, EPSILON) == 1:
        action = rand.choice(6)
        greedy = False
    else:
        action = torch.argmax(values).item()
        greedy = True
    
    return action, values, greedy


def get_next(next_obs, action, reward, old_obs):
    next_obs = process(observation)
    next_obs = numpy.asanyarray(next_obs, dtype=numpy.float32)
    next_obs = torch.from_numpy(next_obs)
    next_state = torch.cat((old_obs[1:], next_obs.unsqueeze(0)))
    new_transition = (old_obs, action, reward, next_state)
    return new_transition


def get_randoms():
    if len(D) < 5:
        return False

    indices = rand.choice(len(D), size=5).tolist()
    randoms = [D[index] for index in indices]
    return randoms

def learn(q_target:Q_Function, q_current:Q_Function, current_transition, transitions=None):

    q_target.zero_grad()

    target = torch.tensor([transition[2] + GAMMA * torch.max(q_target(transition[3])).item() for transition in transitions])
    current = torch.tensor([q_current(transition[0])[transition[1]] for transition in transitions])
    

    loss = q_current.loss(target, current)
    
    loss.requires_grad = True
    loss.backward()
    q_current.optimizer.step()
    q_current.target_counter += 1

def train(episodes):

    for _ in range (episodes):

        input = collect_experience()
        action, values, greedy = e_greedy(input)
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            observation, info = env.reset()
            continue
        
        new_transition = get_next(observation, action, reward, input)
        D.append(new_transition)
        if len(D) < 5:
            continue
        randoms = get_randoms()
        randoms.append(new_transition)
        learn(q_target, q_current, new_transition, randoms)
        if q_current.target_counter == 1000:
            q_target.load_state_dict(q_current.state_dict())
    

def play(n_gamnes):

    for _ in range(n_gamnes):

        obs = env.reset()
        score = 0
        gg = False
        input = collect_experience()
        while not gg:
            
            q_vals = q_current(input)
            action = torch.argmax(q_vals)
            new_obs, reward, terminated, truncated, info = env.step(action)

            score += reward
            obs = new_obs
            new = get_next(obs, action, reward, input)
            input = new[3]

        print("Score:", score) 

train(1000)
play(50)
env.close()

