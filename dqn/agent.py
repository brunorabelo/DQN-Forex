import numpy as np
import random
from collections import namedtuple, deque

from .model import QNetwork, DuelingQNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed=0, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=None,
                 epsilon_delta=None, learning_rate=0.001, batch_size=64, buffer_size=1e5):

        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        assert (epsilon_delta is not None or epsilon_decay is not None), "Epsilon delta and epsilon decay were not defined"

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_start = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.epsilon_delta = epsilon_delta
        self.learning_rate = learning_rate

        self.seed = random.seed(seed)

        self.update_every = 4
        self.tau = 1e-3
        self.batch_size = batch_size
        self.buffer_size = int(buffer_size)

        # Q-Network
        # self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        # self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        # self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Dueling-Q-Network
        self.qnetwork_local = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = DuelingQNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target.load_state_dict(self.qnetwork_local.state_dict())
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def load(self, path):
        self.qnetwork_local.load_state_dict(torch.load(path))

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # update target every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0:

            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

            # self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def act(self, state):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def update_epsilon(self):
        """
        Updates the epsilon used for epsilon-greedy action selection.
        """
        if self.epsilon_delta:
            self.epsilon -= self.epsilon_delta
        else:
            self.epsilon *= self.epsilon_decay
        if self.epsilon < self.epsilon_min:
            self.epsilon = self.epsilon_min

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Used for matrix index operations
        indices = np.arange(self.batch_size)

        self.qnetwork_target.eval()
        self.qnetwork_local.train()

        # Get expected Q values from local model > Shape: (64,1)
        Q_expected = self.qnetwork_local(states).gather(1,
                                                        actions)  # Selects the Q value for the corresponding action made in state

        #### Double Q Learning Steps ####
        # Put local network in testing mode to calculate values
        self.qnetwork_local.eval()

        with torch.no_grad():
            # Get Q values for all actions for each state from local nw > Shape: (64, 4)
            Q_local_next = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1).long()

            # Get index of maximum value for action from Q_local_next > Shape: (64, 1)
            # max_actions = torch.argmax(Q_local_next, dim=1).detach()

            # Get Q values for all actions for each state from target nw > Shape: (64, 4)
            Q_targets_next = self.qnetwork_target(next_states).gather(1, Q_local_next)

        # Put it back to train mode to update gradients and parameters
        self.qnetwork_local.train()

        # Compute Q targets for current states using Double DQN Algorithm
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

    def save(self, path):
        torch.save(self.qnetwork_local.state_dict(), path)


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
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
