import random
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
from model import DQN

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (
            np.array(state, dtype=np.float32),
            np.array(action, dtype=np.int64),
            np.array(reward, dtype=np.float32),
            np.array(next_state, dtype=np.float32),
            np.array(done, dtype=np.float32)
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(
        self, 
        state_dim: int, 
        action_dim: int, 
        hidden_dim: int,
        lr: float, 
        gamma: float,
        tau: float,
        buffer_capacity: int
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.device = DEVICE

        # Policy network (actively trained) and Target network (slowly updated)
        self.policy_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim, hidden_dim).to(self.device)
        
        # Copy initial weights to target network and set target net to evaluation mode
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Adam optimizer for the policy network
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        
        # Experience replay buffer
        self.memory = ReplayBuffer(capacity=buffer_capacity)

    def select_action(self, state: np.ndarray, epsilon: float) -> int:
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self, batch_size: int):
        # Wait until replay buffer has enough samples
        if len(self.memory) < batch_size:
            return

        # Sample a mini-batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Convert numpy arrays to PyTorch tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute Q(s, a) using policy network
        current_q_values = self.policy_net(states).gather(1, actions)

        # Compute target Q-values: R + gamma * max Q_target(s', a') * (1 - done)
        with torch.no_grad():
            max_next_q_values = self.target_net(next_states).max(dim=1)[0].unsqueeze(1)
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute Mean Squared Error (MSE) loss
        loss = F.mse_loss(current_q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradient issues
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

    def soft_update_target_network(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    def save_weights(self, filepath: str = "weights.pth"):
        torch.save(self.policy_net.state_dict(), filepath)

    def load_weights(self, filepath: str = "weights.pth"):
        self.policy_net.load_state_dict(torch.load(filepath, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())