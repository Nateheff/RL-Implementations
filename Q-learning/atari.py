import gymnasium as gym
import ale_py
import torch
import torch.nn as nn
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
env = gym.make("ALE/Pong", render_mode="human")
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
        
    def forward(self, x:torch.Tensor):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = x.flatten()
        x = self.lin(x)
        x = self.out(x)
        return x
        

q = Q_Function(16, 8, 4)


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
    values = q(state)
    if rand.binomial(1, EPSILON) == 1:
        action = rand.choice(6)
        greedy = False
    else:
        action = torch.argmax(values).item()
        greedy = True
    
    return action, values, greedy


def get_randoms():
    if len(D) < 5:
        return False

    indices = rand.choice(len(D), size=5).tolist()
    randoms = [D[index] for index in indices]
    return randoms


for _ in range(10):

    input = collect_experience()
    action, values, greedy = e_greedy(input)
    observation, reward, terminated, truncated, info = env.step(action)
    new_transition = (input, action, reward, process(observation=observation))
    D.append(new_transition)
    randoms = get_randoms()
    target = 0
    if randoms:
        random_targets = []
        for random in randoms:
            value_rand = random[2] + GAMMA * torch.max(q(random[0])).item()
            random_targets.append(value_rand)
        target = max(random_targets)
    else:
        target = reward + GAMMA * torch.max(values).item()

    loss = (target - values[action]) ** 2
    print(loss, values[action], action)
    if terminated or truncated:
        observation, info = env.reset()
    



env.close()

