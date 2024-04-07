import random
import torch
import numpy as np
from collections import deque, namedtuple
from scipy.special import softmax
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Agent():

    def __init__(self, q_network, state_size, action_size, buffer_size, batch_size, optimizer_name, lr, epsilon, gamma, update_every, max_grad_norm, seed):

        ''' Agent Environment Interaction '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        ''' Q-Network '''
        self.qnetwork_local = q_network(state_size, action_size, seed).to(device)
        self.qnetwork_target = q_network(state_size, action_size, seed).to(device)
        self.optimizer = getattr(optim, optimizer_name)(self.qnetwork_local.parameters(), lr=lr)
        
        # self.tau = temperature
        self.eps = epsilon
        self.gamma = gamma
        self.update_every = update_every
        self.max_grad_norm = max_grad_norm

        self.batch_size = batch_size

        ''' Replay memory '''
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)

        ''' Initialize time step (for updating every UPDATE_EVERY steps)           -Needed for Q Targets '''
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):

        ''' Save experience in replay memory '''
        self.memory.add(state, action, reward, next_state, done)

        ''' If enough samples are available in memory, get random subset and learn '''
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

        """ +Q TARGETS PRESENT """
        ''' Updating the Network every 'UPDATE_EVERY' steps taken '''
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:

            self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())

    def act(self, state):

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # ''' Softmax action selection '''
        # return np.random.choice(np.arange(self.action_size), p=softmax(action_values.cpu().data.numpy()[0]/self.tau))

        if random.random() > self.eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))
    
    def act_greedy(self, state):
        
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
            
        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, gamma):
        """ +E EXPERIENCE REPLAY PRESENT """
        states, actions, rewards, next_states, dones = experiences

        ''' Get max predicted Q values (for next states) from target model'''
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        ''' Compute Q targets for current states '''
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        ''' Get expected Q values from local model '''
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        ''' Compute loss '''
        loss = F.mse_loss(Q_expected, Q_targets)

        ''' Minimize the loss '''
        self.optimizer.zero_grad()
        loss.backward()

        ''' Gradiant Clipping '''
        for param in self.qnetwork_local.parameters():
            param.grad.data.clamp_(-self.max_grad_norm, self.max_grad_norm)

        self.optimizer.step()

class DuelingQNetwork1(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """

        super(DuelingQNetwork1, self).__init__()
        self.seed = torch.manual_seed(seed)

        # Common layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # Advantage layer
        self.advantage = nn.Linear(fc2_units, action_size)

        # Value layer
        self.value = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Build a network that maps state -> action values.
        Uses Type-1 Dueling Architecture"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        advantage = self.advantage(x)
        value = self.value(x)

        return value + advantage - advantage.mean()

class DuelingQNetwork2(DuelingQNetwork1):

  def forward(self, state):
      """Build a network that maps state -> action values.
      Uses Type-2 Dueling Architecture"""
      x = F.relu(self.fc1(state))
      x = F.relu(self.fc2(x))

      advantage = self.advantage(x)
      value = self.value(x)

      return value + advantage - advantage.max()