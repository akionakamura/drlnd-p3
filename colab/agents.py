import numpy as np
import random
import copy

import torch
import torch.nn.functional as F
import torch.optim as optim

from memory import ReplayBuffer
from models import Actor, Critic


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


DEFAULT_GAMMA = 0.99            # discount factor
DEFAULT_TAU = 2e-1              # for soft update of target parameters
DEFAULT_LR_ACTOR = 1e-4         # learning rate of the actor 
DEFAULT_LR_CRITIC = 3e-4        # learning rate of the critic
DEFAULT_BATCH_SIZE = 4096


class SingleAgentDDPG():
    def __init__(
        self,
        state_size: int,
        action_size: int,
        gamma: float = DEFAULT_GAMMA,
        learning_rate_actor: float = DEFAULT_LR_ACTOR,
        learning_rate_critic: float = DEFAULT_LR_CRITIC,
        tau: float = DEFAULT_TAU,
        learn_step: int = 1,
        sync_step: int = 1,
        epsilon_start: float = 1.0,
        min_epsilon: float = 0.01,
        epsilon_decay: float = 0.999,
        batch_size: int = DEFAULT_BATCH_SIZE
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.learning_rate_actor = learning_rate_actor
        self.learning_rate_critic = learning_rate_critic
        self.tau = tau
        self.learn_step = learn_step
        self.sync_step = sync_step
        self.epsilon = self.epsilon_start = epsilon_start
        self.min_epsilon = min_epsilon
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.actor_local = Actor(self.state_size, self.action_size).to(device)
        self.actor_target = Actor(self.state_size, self.action_size).to(device)
        self.actor_optimizer = optim.Adam(
            self.actor_local.parameters(), lr=self.learning_rate_actor)

        self.critic_local = Critic(self.state_size, self.action_size).to(device)
        self.critic_target = Critic(self.state_size, self.action_size).to(device)
        self.critic_optimizer = optim.Adam(
            self.critic_local.parameters(), lr=self.learning_rate_critic)

        self.replay_buffer = ReplayBuffer(batch_size=self.batch_size)
        self.noise = OUNoise((action_size, ), 42)
        self.t = 0

        self.gamma_tensor = torch.tensor(self.gamma, requires_grad=False).float().to(device)
        self.one_tensor = torch.tensor(1, requires_grad=False).float().to(device)
    
    def reset(self):
        self.noise.reset()
    
    def act(self, states, add_noise=True):
        states = torch.from_numpy(np.vstack(states)).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()

        if add_noise:
            action += self.noise.sample() * self.epsilon

        return np.clip(action, -1, 1)
    
    def step(self, state, action, reward, next_state, done):
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.t += 1

        # If a multiple of `learn_step`, it is time to learn.
        if self.t % self.learn_step == 0:
            self.learn()

    def learn(self):
        states, actions, rewards, next_states, dones = self.replay_buffer\
            .sample()
        
        # Convert everything to PyTorch tensors.
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).float().to(device)
        rewards = torch.from_numpy(np.hstack(rewards)).float().unsqueeze(dim=1).to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.hstack(dones).astype(int)).float()\
            .unsqueeze(dim=1).to(device)

        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states).detach()
        Q_targets_next = self.critic_target(next_states, actions_next).detach()
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.gamma_tensor * Q_targets_next * (self.one_tensor - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        if self.t % self.sync_step == 0:
            self.soft_update(self.critic_local, self.critic_target, self.tau)
            self.soft_update(self.actor_local, self.actor_target, self.tau)
    
    def episode_finished(self):
        """At the end of an episode, update parameters"""
        # Update epsilon value.
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)
    
    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

    def save(self, path_idx=0, id=0):
        torch.save(self.critic_local.state_dict(), f"./models/{path_idx}/critic-{id}.pth")
        torch.save(self.actor_local.state_dict(), f"./models/{path_idx}/actor-{id}.pth")
    
    def load(self, path_idx=0, id=0):
        self.actor_local.load_state_dict(torch.load(f"./models/{path_idx}/actor-{id}.pth"))
        self.actor_target.load_state_dict(torch.load(f"./models/{path_idx}/actor-{id}.pth"))

        self.critic_local.load_state_dict(torch.load(f"./models/{path_idx}/critic-{id}.pth"))
        self.critic_target.load_state_dict(torch.load(f"./models/{path_idx}/critic-{id}.pth"))

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.uniform(size=x.shape)
        self.state = x + dx
        return self.state
