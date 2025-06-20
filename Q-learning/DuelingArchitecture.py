import torch
import torch.nn as nn
import torch.optim as optim
from utils import *

class Dueling_DQN(nn.Module):
    def __init__(self, kernels, kernel_dim, stride):
        super().__init__()
        #Shared Convolution Encoder
        self.conv1 = nn.Conv2d(in_channels=4, out_channels=kernels, kernel_size=kernel_dim, stride=stride)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=32, kernel_size=4, stride=2)

        #! fully-connected layer
        self.lin1 = nn.Linear(32*9*9, out_features=512)

        self.value_stream = nn.Linear(512, out_features=1)
        self.advantage_stream = nn.Linear(512, out_features=6)
        

        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=0.01)
        self.target_counter = 0

    def forward(self, input):

        x = self.conv1(input)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = x.view(x.size()[0], -1)


        x = self.lin1(x)
        x = self.relu(x)

        value = self.value_stream(x)
        advantages = self.advantage_stream(x)

        return value, advantages

    def aggreagate(self, values, advantages):
        Q = values + (advantages - advantages.mean(dim=1,keepdim=True))
        return Q


q_current = Dueling_DQN(16, 8, 4)
q_target = Dueling_DQN(16, 8, 4)


def choose_action(state:torch.Tensor):
    state = state.unsqueeze(0)
    values, advantages = q_current(state)
    if rand.binomial(1, EPSILON) == 1:
        action = rand.choice(6)
        greedy = False
    else:
        action = torch.argmax(advantages).item()
        greedy = True
    
    return action

def learn_DuelingDDQN(q_target:Dueling_DQN, q_current:Dueling_DQN, transitions=None):
    """

    """
    q_target.zero_grad()
    check = transitions[0]
    current_batch = create_batches(transitions, 0)
    target_batch = create_batches(transitions, 3)

    target_values, target_advantages = q_target(target_batch)
    current_values, current_advantages = q_current(current_batch)
    

    Q_targets = q_target.aggreagate(target_values, target_advantages)
    Q_currents = q_current.aggreagate(current_values, current_advantages)

    target = None

    loss = q_current.loss(Q_targets, Q_currents)
    
    # loss.requires_grad = True
    loss.backward()
    q_current.optimizer.step()
    q_current.target_counter += 1
    if(q_current.target_counter % 100 == 0):
        print(loss)


def train(episodes):

    for i in range (episodes):
        
        input = collect_experience(env)
        action = choose_action(input)
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
        learn_DuelingDDQN(q_target, q_current, randoms)
        if q_current.target_counter == 1000:
            q_target.load_state_dict(q_current.state_dict())


train(5000)