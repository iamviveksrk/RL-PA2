import random
import torch
import numpy as np
from collections import deque, namedtuple
from scipy.special import softmax
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch import distributions

eps = np.finfo(np.float32).eps.item()

class PolicyNetwork(nn.Module):

    def __init__(self, state_size, action_size, seed, fc1_units=128):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.dropout = nn.Dropout(p=0.6)

        self.actions = nn.Linear(fc1_units, action_size)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = F.relu(x)
        action_scores = self.actions(x)
        return F.softmax(action_scores, dim=1)

def select_action(state, policy):
    state = torch.from_numpy(state).float().unsqueeze(0)
    probs = policy(state)
    m = distributions.Categorical(probs)
    action = m.sample()
    policy.saved_log_probs.append(m.log_prob(action))
    return action.item()


def finish_episode(policy, optimizer, gamma):
    R = 0
    policy_loss = []
    returns = deque()
    for r in policy.rewards[::-1]:
        R = r + gamma * R
        returns.appendleft(R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + eps)
    for log_prob, R in zip(policy.saved_log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]

def run_reinforce(state_size, action_size, env, seed, fc1_units=128, optimizer_name='Adam', lr=1e-2, n_episodes=500, gamma=0.99):

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