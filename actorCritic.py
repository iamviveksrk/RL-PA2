import random
import torch
import numpy as np
from collections import deque, namedtuple
from scipy.special import softmax
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch import distributions

import warnings
warnings.filterwarnings("ignore")

eps = np.finfo(np.float32).eps.item()
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

class PolicyNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)

        self.actions = nn.Linear(fc1_units, action_size)
        self.value = nn.Linear(fc1_units, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        action_probs = F.softmax(self.actions(x), dim=1)
        state_value = self.value(x)
        
        return action_probs, state_value

def select_action(state, model):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs, state_value = model(state)
    m = distributions.Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode(model, optimizer, gamma):
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    returns = deque()
    for r in model.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        advantage = R - value.item()
        policy_losses.append(-log_prob * advantage)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))

    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]

def run_actorCritic(state_size, action_size, env, seed, fc1_units=128, optimizer_name='Adam', lr=1e-2, n_episodes=500, gamma=0.99):

    policy = PolicyNetwork(state_size, action_size, seed, fc1_units=128)
    optimizer = getattr(optim, optimizer_name)(policy.parameters(), lr=lr)

    episode_rewards = []
    scores_window = deque(maxlen=100)

    for i_episode in range(1, n_episodes+1):
        state, _ = env.reset()
        ep_reward = 0
        for t in range(1, 10000):  # Don't infinite loop while learning
            action = select_action(state, policy)
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            policy.rewards.append(reward)
            ep_reward += reward
            if done:
                break
        
        episode_rewards.append(ep_reward)
        scores_window.append(ep_reward)

        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")

        finish_episode(policy, optimizer, gamma)

        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        
    return np.array(episode_rewards)