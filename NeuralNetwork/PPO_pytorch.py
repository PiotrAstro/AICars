import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class PPO:
    def __init__(self, state_length, state_features, action_dim, entropy_factor=0.01, epsilon=0.1,
                 gamma=0.99, lamda_tradeof=0.95, actor_learning_rate=0.003, critic_learning_rate=0.0003,
                 actor_epochs=10, critic_epochs=10, batch_size=32):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            print("Using GPU")
        else:
            print("Using CPU")
        self.state_length = state_length
        self.state_features = state_features
        self.action_dim = action_dim
        self.entropy_factor = entropy_factor
        self.epsilon = epsilon
        self.gamma = gamma
        self.lamda_tradeof = lamda_tradeof
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate
        self.actor_epochs = actor_epochs
        self.critic_epochs = critic_epochs
        self.batch_size = batch_size

        self.model = None

        self.reset_with_new_weights()
        device_check = next(self.model.parameters()).device
        print(device_check)

    def reset_with_new_weights(self):
        self.model = self.PPO_model(self.state_length, self.state_features, self.action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.model.critic.parameters(), lr=self.critic_learning_rate)
        self.actor_optimizer = optim.Adam(list(self.model.combined.parameters()) + list(self.model.actor.parameters()),
                                          lr=self.actor_learning_rate)
        self.actor_optimizer_categorical = optim.Adam(
            list(self.model.combined.parameters()) + list(self.model.actor.parameters()), lr=self.actor_learning_rate)
        self.critic_loss = nn.MSELoss()
        self.actor_loss = self.Actor_loss()
        self.actor_categorical_loss = nn.CrossEntropyLoss()

    def save_weights(self, file_name=""):
        path = "model_weights/PPO/" + file_name + ".pth"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'actor_optimizer_categorical_state_dict': self.actor_optimizer_categorical.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
        }, path)
        return path

    def load_weights(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.actor_optimizer_categorical.load_state_dict(checkpoint['actor_optimizer_categorical_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

    class PPO_model(nn.Module):
        def __init__(self, state_length, state_features, action_dim):
            super(PPO.PPO_model, self).__init__()
            self.combined = nn.Sequential(
                nn.Linear(state_length * state_features, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Flatten()
            )

            self.actor = nn.Sequential(
                nn.Linear(32, action_dim),
                nn.Softmax(dim=1)
            )

            self.critic = nn.Sequential(
                nn.Linear(32, 1)
            )
        def forward(self, x):
            x = self.combined(x)
            probs = self.actor(x)
            value = self.critic(x)
            return probs, value

    class Actor_loss(nn.Module):
        def __init__(self):
            super(PPO.Actor_loss, self).__init__()

        def forward(self, y_pred, y_true, advantage, old_prediction, epsilon, entropy_factor):
            prob = y_true * y_pred
            old_prob = y_true * old_prediction
            r = prob / (torch.clamp(old_prob, min=1e-10))
            clipped_r = torch.clamp(r, 1 - epsilon, 1 + epsilon)
            entropy_bonus = entropy_factor * (prob * torch.log(torch.clamp(prob, min=1e-10)))
            return -torch.mean(torch.min(r * advantage, clipped_r * advantage) + entropy_bonus)

    def get_action_probs(self, state):
        state = torch.tensor([state], dtype=torch.float32).to(self.device)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        return action_probs[0].detach().cpu().numpy()

    def train(self, states, actions_indexes, rewards, action_probabilities=None, expected_rewards=None, advantages=None):
        if expected_rewards is None:
            expected_rewards = self.get_expected_rewards(states, rewards)

        if advantages is None:
            advantages = self.get_Advantages(states, rewards)



        states = torch.from_numpy(states).float().to(self.device)
        expected_rewards = torch.from_numpy(expected_rewards).float().to(self.device)
        expected_rewards = expected_rewards.view(-1, 1).to(self.device)
        advantages = torch.from_numpy(advantages).float().to(self.device)
        advantages = advantages.view(-1, 1)

        if action_probabilities is None:
            with torch.no_grad():
                action_probabilities, _ = self.model(states)
        else:
            action_probabilities = torch.from_numpy(action_probabilities).float().to(self.device)

        # Convert actions to one-hot encoding
        actions_indexes = torch.from_numpy(np.array(actions_indexes)).long().to(self.device)
        actions_onehot = F.one_hot(actions_indexes, self.action_dim).float()

        # Create a TensorDataset and DataLoader
        dataset = TensorDataset(states, expected_rewards, actions_onehot, advantages, action_probabilities)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        for _ in range(self.actor_epochs):
            for batch_states, _, batch_actions_onehot, batch_advantages, batch_old_prediction in dataloader:
                self.actor_optimizer.zero_grad()
                current_probabilities, _ = self.model(batch_states)
                loss = self.actor_loss(current_probabilities, batch_actions_onehot, batch_advantages, batch_old_prediction, self.epsilon, self.entropy_factor)
                loss.backward()
                self.actor_optimizer.step()

        for _ in range(self.critic_epochs):
            for batch_states, batch_expected_rewards, _, _, _ in dataloader:
                self.critic_optimizer.zero_grad()
                _, values = self.model(batch_states)
                loss = self.critic_loss(values, batch_expected_rewards)
                loss.backward()
                self.critic_optimizer.step()

    def get_expected_rewards(self, states, rewards, use_last_value=True):
        state = torch.from_numpy(np.array([states[-1]])).float().to(self.device)
        with torch.no_grad():
            _, value = self.model(state)
        value = value.detach().cpu().numpy()[0][0] if use_last_value else 0.0
        expected_rewards = np.zeros_like(rewards)
        for t in reversed(range(len(rewards))):
            value = rewards[t] + self.gamma * value
            expected_rewards[t] = value
        return expected_rewards

    def get_Advantages(self, states, rewards):
        states = torch.from_numpy(np.array(states)).float().to(self.device)
        with torch.no_grad():
            _, values = self.model(states)
        values = values.detach().cpu().numpy()
        advantages = np.zeros_like(rewards)
        gae = 0
        next_values = values[-1][0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * next_values - values[t][0]
            gae = delta + self.gamma * self.lamda_tradeof * gae
            advantages[t] = gae
            next_values = values[t][0]
        return np.array(advantages)
