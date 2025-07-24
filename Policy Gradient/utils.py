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

def get_episode(policy, num_left, lstm=False):
    
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
            if lstm:
                _, probs, _, _ = policy(current_state)
            else:
                _, probs = policy(current_state)
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

def get_episode_GRPO(policy, num_left, G):
    global env
    obs, info = env.reset()
    state = process(obs)
    current_state = torch.from_numpy(state)

    done = False

    episode_states = []
    episode_actions = []
    episode_rewards = []
    epsiode_values = []
    episode_probs = []

    t=0
    while not done and t < num_left:
        with torch.no_grad():
            current_state = current_state.unsqueeze(0).unsqueeze(0)

            value, probs = policy(current_state)
            
            dist = torch.distributions.Categorical(probs)
            env_start = env
            state_actions = []
            state_rewards = []
            for i in range(G):
                #We take G samples at each state and later use the average of these samples' rewards
                #to calculate advantage
                env = env_start
                action = dist.sample()
                next_obs, reward, terminated, truncated, info = env.step(action)
                state_actions.append(action)
                state_rewards.append(reward)
            episode_actions.append(state_actions)
            episode_rewards.append(state_rewards)
            probs = probs.repeat(G,1)

        next_state = process(next_obs)
        next_state = torch.from_numpy(next_state)
        
        episode_states.append(current_state.squeeze(0))
        episode_probs.append(probs)
        epsiode_values.append(value)

        current_state = next_state
        t += 1
        done = terminated or truncated

    return episode_states, episode_actions, episode_rewards, epsiode_values, episode_probs



def process_batch(batch, policy, batch_size, episode_boundaries, hidden_size=0):
    if hidden_size:
        hidden_state = torch.zeros((batch_size, hidden_size))
        cell_state = torch.zeros(batch_size, hidden_size)

    values = []
    probs = []

    prev_boundary = 0
    for boundary in episode_boundaries:
        episode_states = batch[prev_boundary: boundary]
        if hidden_size:
            episode_values, episode_probs, _, _ = policy(episode_states)
        else: 
            episode_values, episode_probs = policy(episode_states)
        
        values.append(episode_values)
        probs.append(episode_probs)

        prev_boundary = boundary

    return torch.cat(values), torch.cat(probs)
        

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


def get_batches_GAE(batch_size, policy, hidden_size=None):

    batch_actions = []
    batch_advantages = []
    
    
    batch_rewards = []
    batch_states = []
    batch_returns = []
    episode_boundaries = []

    current_episode = 0

    while len(batch_states) < batch_size:
        episode_states, episode_actions, episode_rewards = get_episode(policy, batch_size - len(batch_states), hidden_size)
        batch_states.extend(episode_states)
        batch_actions.extend(episode_actions)
        batch_rewards.extend(episode_rewards)
        episode_boundaries.append(len(batch_states))

    batch = torch.stack(batch_states)
    batch_actions = torch.cat(batch_actions)

    batch_values, batch_probs = process_batch(batch, policy, batch_size, episode_boundaries, hidden_size)
    dist = torch.distributions.Categorical(batch_probs)
    batch_log_probs = dist.log_prob(batch_actions)
    batch_rewards = torch.tensor(batch_rewards)

    prev_boundary = 0
    for boundary in episode_boundaries:
        # Get episode data
        episode_rewards = batch_rewards[prev_boundary:boundary]
        episode_values = batch_values[prev_boundary:boundary]
        episode_states = batch_states[prev_boundary:boundary]
        
        # Compute returns for this episode
        episode_returns = []
        discounted_return = 0

        for r in reversed(episode_rewards):
            discounted_return = r + GAMMA * discounted_return
            episode_returns.append(discounted_return)
        episode_returns.reverse()

        episode_advantages = []
        gae = 0
        
        # Get next value (0 for terminal state)
        next_value = 0  # Terminal state value

        for i in range(len(episode_rewards)):
            if i == len(episode_rewards) - 1:
                #last step
                current_td = episode_rewards[i] - episode_values[i]
            else:
                current_td = episode_rewards[i] + GAMMA * episode_values[i + 1] - episode_values[i]

            gae = current_td + GAMMA*LAMBDA * gae
            batch_advantages.insert(0, gae)
            next_value = batch_values[i]

        # Add to batch lists
        batch_returns.extend(episode_returns)
        batch_advantages.extend(episode_advantages)
        
        prev_boundary = boundary

    batch_advantages = torch.tensor(batch_advantages)
    batch_returns = torch.tensor(batch_returns)
    return batch_values, batch_actions, batch_advantages, batch_log_probs, batch_returns


def get_batches_GRPO(batch_size, policy):

    batch_actions = []
    batch_advantages = []
    
    
    batch_rewards = []
    batch_states = []
    batch_returns = []
    batch_values = []
    batch_probs = []
    episode_boundaries = []

    current_episode = 0

    while len(batch_states) < batch_size:
        episode_states, episode_actions, episode_rewards, episode_values, episode_probs = get_episode_GRPO(policy, batch_size - len(batch_states), 10)
        batch_states.extend(episode_states)
        batch_actions.extend(episode_actions)
        batch_rewards.extend(episode_rewards)
        batch_values.extend(episode_values)
        batch_probs.extend(episode_probs)
        episode_boundaries.append(len(batch_states))

    
    batch_actions = torch.tensor(batch_actions)
    batch_values = torch.tensor(batch_values)
    batch_probs = torch.stack(batch_probs)
    dist = torch.distributions.Categorical(batch_probs)
    batch_log_probs = dist.log_prob(batch_actions)
    batch_rewards = torch.tensor(batch_rewards)

    prev_boundary = 0
    for boundary in episode_boundaries:
        # Get episode data
        episode_rewards = batch_rewards[prev_boundary:boundary]
        episode_values = batch_values[prev_boundary:boundary]
        episode_states = batch_states[prev_boundary:boundary]

        episode_advantages = []
        gae = 0
        
        episode_advantages = (episode_rewards - episode_rewards.mean(dim=-1, keepdim=True)) / (episode_rewards.std(dim=-1, keepdim=True) + 1e-8)

        # Add to batch lists
        batch_advantages.extend(episode_advantages)
        
        prev_boundary = boundary

    batch_advantages = torch.stack(batch_advantages)
    batch_states = torch.stack(batch_states)
    return batch_states, batch_actions, batch_advantages, batch_log_probs