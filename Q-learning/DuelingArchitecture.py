import torch
import torch.nn as nn
import torch.optim as optim
from utils import *
from PrioritizedReplay import PrioritizedMemory



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
        
        self.lr = 1e-4
        self.loss = nn.MSELoss()
        self.optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
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

def get_DDQN_action(state):
    values, advantages = q_current(state)
    action = torch.argmax(advantages).item()

    return action

def choose_action(state:torch.Tensor):
    state = state.unsqueeze(0)
    
    if rand.binomial(1, EPSILON) == 1:
        action = rand.choice(6)
        
    else:
        action = get_DDQN_action(state)
        
    
    return action

def parse_transitions(transitions):
    #We need to return 3 tensors each containing the values of each transition at the respective index
    actions, rewards = ([transition[i+1] for transition in transitions] for i in range(2))
    next_states = torch.stack([t[3] for t in transitions])
    
    return torch.tensor(actions), torch.tensor(rewards), next_states



def learn_DuelingDDQN(q_target:Dueling_DQN, q_current:Dueling_DQN, transitions=None):

    q_target.zero_grad()
    check = transitions[0]
    current_batch = create_batches(transitions, 0)
    target_batch = create_batches(transitions, 3)

    target_values, target_advantages = q_target(target_batch)
    current_values, current_advantages = q_current(current_batch)
    

    Q_targets = q_target.aggreagate(target_values, target_advantages)
    Q_currents = q_current.aggreagate(current_values, current_advantages)



    loss = q_current.loss(Q_targets, Q_currents)
    

    loss.backward()
    q_current.optimizer.step()
    q_current.target_counter += 1


def learn_DuelingDDQN_prioritized(q_target:Dueling_DQN, q_current:Dueling_DQN, actions, rewards, next_states, weights, transitions=None):


    
    current_batch = create_batches(transitions, 0)
    Q_streams= q_current(current_batch)
    #Q_values of all actions for each transition in batch
    Q_currents = q_current.aggreagate(*Q_streams)

    #Q_values of chosen actions in batch
    Q_chosen = Q_currents.gather(1, actions.unsqueeze(1)).squeeze() 
    
    with torch.no_grad():
        next_q = q_current(next_states)
        best_actions = q_current.aggreagate(*next_q).argmax(1)
        target_q = q_target.aggreagate(*q_target(next_states))
        q_targets = rewards + GAMMA * target_q.gather(1, best_actions.unsqueeze(1)).squeeze()

    td_errors = q_targets - Q_chosen
    loss = (weights * td_errors.pow(2)).mean()
    
    loss.backward()
    q_current.optimizer.step()
    
    return td_errors




def get_Q(model: Dueling_DQN, observation, action):
    model.zero_grad()
    values, advantages = model(observation)

    Q_final = model.aggreagate(values, advantages).squeeze()
    return Q_final[action].item()
    


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



def train_prioritized(episodes):
    memory = PrioritizedMemory(200)
    
    for i in range (episodes):
        
        input = collect_experience(env)
        action = choose_action(input)
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()
            continue
        
        new_transition = get_next(observation, action, reward, input)

        if memory.count <= memory.batch_size:
            memory.add(new_transition, 1)
            continue

        randoms, indices = memory.get()
        randoms.append(new_transition)

        indices = torch.cat((indices, torch.tensor([memory.next_idx])))
        actions, rewards, next_states = parse_transitions(randoms)

        weights = memory.weights(indices)
        weights[4] = 0
        weights = torch.tensor(weights)

        priorities = learn_DuelingDDQN_prioritized(q_target, q_current, actions, rewards, next_states, weights,randoms)

        for i,idx in enumerate(indices):
            memory.priorities[idx] = abs(priorities[i])

        memory.add(new_transition, abs(priorities[4]))

        if q_current.target_counter == 100:
            q_target.load_state_dict(q_current.state_dict())
            q_current.target_counter = 0

train_prioritized(500)





"""
        for random, index in zip(randoms, indices):
            next_state = random[3].unsqueeze(0)
            current_state = random[0].unsqueeze(0)
            weight = memory.weights(index)
            value = random[2] + GAMMA * get_Q(q_target, next_state,get_DDQN_action(next_state)) - get_Q(q_current, current_state, random[1])
            value = abs(value) + 1e-8

            memory.update(index, value)
            
            loss = weight * (value ** 2)
            loss.backward()
                
"""           