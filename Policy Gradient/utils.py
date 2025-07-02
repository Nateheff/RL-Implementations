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

batch_size = 4

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


def get_batches_TRPO(batch_size, policy):

    states = []
    actions = []
    rewards = []
    log_probs = []
    advantages = []


    while len(states) < batch_size:
        obs, info = env.reset()
        state = process(obs)
        
        initial_state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
        done = False

        episode_rewards = []
        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_q_vals = []

        while not done:
            with torch.no_grad():
                q_vals, probs = policy(initial_state)
                
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

            new_obs, reward, terminated, truncated, info = env.step(action.item())
            
            next_state = process(new_obs)
            next_state = torch.from_numpy(next_state).unsqueeze(0).unsqueeze(0)

            episode_states.append(next_state.squeeze(0))
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)
            episode_q_vals.append(q_vals.squeeze(0))

            initial_state = next_state
            done = terminated or truncated

        total_return = 0
        discounted_rewards = []
        for r in reversed(episode_rewards):
            total_return = r + GAMMA * total_return
            discounted_rewards.insert(0, total_return)
        
        for i in range(len(episode_rewards)):
            q = episode_q_vals[i]
            a = episode_actions[i]
            advantage = discounted_rewards[i] - q[a]
            advantages.append(advantage)

        states.extend(episode_states)
        actions.extend(episode_actions)
        log_probs.extend(episode_log_probs)
        rewards.extend([total_return]*len(episode_rewards))

    states = states[:batch_size]
    actions = actions[:batch_size] 
    advantages = advantages[:batch_size]
    log_probs = log_probs[:batch_size]
    rewards = rewards[:batch_size]

    return (torch.stack(states), torch.tensor(actions), torch.tensor(advantages), torch.stack(log_probs), rewards)


def get_batches_GAE(batch_size, policy):

    states = []
    actions = []
    returns = []
    log_probs = []
    advantages = []


    while len(states) < batch_size:
        obs, info = env.reset()
        state = process(obs)
        
        initial_state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
        done = False

        episode_rewards = []
        episode_states = []
        episode_actions = []
        episode_log_probs = []
        episode_vals = []

        while not done:
            current_state = initial_state
            with torch.no_grad():
                vals, probs = policy(current_state)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            
            new_obs, reward, terminated, truncated, info = env.step(action.item())
            
            next_state = process(new_obs)
            next_state = torch.from_numpy(next_state).unsqueeze(0).unsqueeze(0)

            episode_states.append(current_state.squeeze(0))
            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)
            episode_vals.append(vals.squeeze(0))

            current_state = next_state
            done = terminated or truncated

        episode_rewards = torch.tensor(episode_rewards)
        episode_vals = torch.stack(episode_vals)

        episode_returns = []
        discounted_return = 0
        for r in reversed(episode_rewards):
            discounted_return = r + GAMMA * discounted_return
            episode_returns.append(discounted_return)

        episode_returns = torch.tensor(episode_returns)

        episode_advantages = []
        gae = 0

        with torch.no_grad():
            final_value, _ = policy(current_state)
            final_value = final_value.squeeze(0)


        next_value = final_value
        for i in reversed(range(len(episode_rewards))):
            if i == len(episode_rewards) - 1:
                #last step
                current_td = episode_rewards[i] - episode_vals[i]
            else:
                current_td = episode_rewards[i] + GAMMA * next_value - episode_vals[i]
            
            gae = current_td + GAMMA * LAMBDA * gae
            episode_advantages.insert(0, gae)
            next_value = episode_vals[i]

        episode_advantages = torch.tensor(episode_advantages)

        states.extend(episode_states)
        actions.extend(episode_actions)
        log_probs.extend(episode_log_probs)
        advantages.extend(episode_advantages)
        returns.extend(episode_returns)

    states = states[:batch_size]
    actions = actions[:batch_size] 
    advantages = advantages[:batch_size]
    log_probs = log_probs[:batch_size]
    returns = returns[:batch_size]

    return (torch.stack(states), torch.tensor(actions), torch.stack(advantages), torch.stack(log_probs), torch.stack(returns))
