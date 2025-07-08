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



def process(observation):

    img = Image.fromarray(observation)
    grayscale = img.convert('L')
    down = grayscale.resize((DOWN,CROP), resample=Image.BILINEAR)
    cropped = down.crop((CROP_LEFT,0, CROP_LEFT + CROP, CROP))
    
    return numpy.array(cropped, dtype=numpy.float32) / 255.0

def get_batches_ACKTR(batch_size, policy, hidden_state, cell_state):

    states = []
    actions = []
    returns = []
    log_probs = []
    advantages = []
    values = []


    while len(values) < batch_size:
        obs, info = env.reset()
        state = process(obs)
        
        initial_state = torch.from_numpy(state).unsqueeze(0).unsqueeze(0)
        done = False

        episode_rewards = []
        episode_actions = []
        episode_log_probs = []
        episode_vals = []

        while not done:
            current_state = initial_state
            with torch.no_grad():
                val, probs, new_hidden, new_cell = policy(current_state, hidden_state, cell_state)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            
            new_obs, reward, terminated, truncated, info = env.step(action.item())
            
            next_state = process(new_obs)
            next_state = torch.from_numpy(next_state).unsqueeze(0).unsqueeze(0)

            episode_actions.append(action)
            episode_log_probs.append(log_prob)
            episode_rewards.append(reward)
            episode_vals.append(val.squeeze(0))

            current_state = next_state
            hidden_state = new_hidden
            cell_state = new_cell
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
            final_value, _, _, _ = policy(current_state, hidden_state, cell_state)
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

        values.extend(episode_vals)
        actions.extend(episode_actions)
        log_probs.extend(episode_log_probs)
        advantages.extend(episode_advantages)
        returns.extend(episode_returns)

    values = values[:batch_size]
    actions = actions[:batch_size] 
    advantages = advantages[:batch_size]
    log_probs = log_probs[:batch_size]
    returns = returns[:batch_size]

    return (torch.stack(values), torch.tensor(actions), torch.stack(advantages), torch.stack(log_probs), torch.stack(returns))