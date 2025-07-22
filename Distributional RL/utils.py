import torch
import numpy.random as rand
from PIL import Image
import numpy
import gymnasium as gym
import ale_py
from collections import deque

HEIGHT = 160
WIDTH = 210
DOWN = 110
CROP = 84
CROP_LEFT = (DOWN - CROP) / 2

EPSILON = 0.1
LAMBDA = 0.95
GAMMA = 0.99

batch_size = 5

gym.register_envs(ale_py)
env = gym.make("ALE/Pong")
observation, info = env.reset(seed=42)

D = deque(maxlen=50)

def process(observation):

    img = Image.fromarray(observation)
    grayscale = img.convert('L')
    down = grayscale.resize((DOWN,CROP), resample=Image.BILINEAR)
    cropped = down.crop((CROP_LEFT,0, CROP_LEFT + CROP, CROP))
    
    return numpy.array(cropped, dtype=numpy.float32) / 255.0


def collect_experience(env):

    history = []
    observation, info = env.reset(seed=42)

    for _ in range(4):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        history.append(process(observation))

    n_hist = numpy.asanyarray(history, dtype=numpy.float32)
    input = torch.from_numpy(n_hist) #input is an 84 x 84 x 4 tensor
    return input


def create_batches(transitions, index):
    try:
        batch = torch.stack([transition[index] for transition in transitions])
        return batch
    except Exception as e:
        print(e, index)
        for transition in transitions:
            print("TRANS: ",transition)
            print("THIRDS: ", transition[3])
        
        

def e_greedy(q_current, state, epsilon=None):
    if epsilon:
        EPSILON = epsilon
    
    values = q_current(state)
    if rand.binomial(1, EPSILON) == 1:
        action = rand.choice(6)
        greedy = False
    else:
        action = torch.argmax(values).item()
        greedy = True
    
    return action, values, greedy


def get_next(next_obs, action, reward, old_obs):
    next_obs = process(next_obs)
    next_obs = numpy.asanyarray(next_obs, dtype=numpy.float32)
    next_obs = torch.from_numpy(next_obs)
    next_state = torch.cat((old_obs[1:], next_obs.unsqueeze(0)))
    new_transition = (old_obs, action, reward, next_state)
    return new_transition


def get_randoms(D):
    if len(D) < batch_size:
        return False

    indices = rand.choice(len(D), size=batch_size).tolist()
    randoms = [D[index] for index in indices]
    return randoms

def get_episode(policy, num_left):
    
    obs, info = env.reset()
    state = process(obs)
    current_state = torch.from_numpy(state)

    done = False

    episode_states = []
    episode_actions = []
    episode_rewards = []

    t=0
    while not done and t < num_left:
        with torch.no_grad():
            current_state = current_state.unsqueeze(0).unsqueeze(0)
            _, probs, _, _ = policy(current_state)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        
        next_obs, reward, terminated, truncated, info = env.step(action)

        next_state = process(next_obs)
        next_state = torch.from_numpy(next_state)
        
        episode_states.append(current_state.squeeze(0))
        episode_actions.append(action)
        episode_rewards.append(reward)

        current_state = next_state
        t += 1
        done = terminated or truncated

    return episode_states, episode_actions, episode_rewards



def process_batch(batch, policy, batch_size, episode_boundaries, hidden_size):
    
    hidden_state = torch.zeros(batch_size, hidden_size)
    cell_state = torch.zeros(batch_size, hidden_size)

    values = []
    probs = []

    prev_boundary = 0
    for boundary in episode_boundaries:
        episode_states = batch[prev_boundary: boundary]
        
        episode_values, episode_probs, _, _ = policy(episode_states)

        values.append(episode_values)
        probs.append(episode_probs)

        prev_boundary = boundary

    return torch.cat(values), torch.cat(probs)
        