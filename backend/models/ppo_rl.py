import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger

class PPOPreferenceLearner(nn.Module):
    """
    Proximal Policy Optimization (PPO) RL model for Adapting to User Preferences.
    It takes user features (num journeys, time of day, delays, etc.) and outputs 
    a continuous policy action [toll_aversion, highway_pref, variance_tolerance].
    """
    def __init__(self, state_dim=5, action_dim=3):
        super(PPOPreferenceLearner, self).__init__()
        # Actor network (Policy)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim),
            nn.Sigmoid()  # Bound preferences between 0 and 1
        )
        # Critic network (Value)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=1e-3)

    def forward(self, state):
        action_probs = self.actor(state)
        state_value = self.critic(state)
        return action_probs, state_value

    def update_policy(self, states, actions, rewards, next_states):
        """
        A simplified PPO update step.
        In a full implementation, this uses clipped surrogate loss on advantage estimations.
        """
        logger.info("Running PPO RL policy update step for user preferences...")
        
        state_tensor = torch.tensor(states, dtype=torch.float32)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
        
        self.optimizer.zero_grad()
        predicted_actions, values = self(state_tensor)
        
        # Simple Advantage actor-critic proxy for the demo
        advantages = rewards_tensor - values.squeeze()
        
        # Actor loss (proxy for PPO clipped loss)
        actor_loss = - (predicted_actions.mean(dim=1) * advantages.detach()).mean()
        
        # Critic loss
        critic_loss = (advantages ** 2).mean()
        
        loss = actor_loss + 0.5 * critic_loss
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

# Global explicit singleton instance for the background training and inference API
ppo_agent = PPOPreferenceLearner()
