import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from itertools import count
from collections import deque,namedtuple
import random
from copy import deepcopy
import gymnasium as gym
import matplotlib.pyplot as plt
import os
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reinforcement Learning')
    parser.add_argument('-c', '--checkpoint', action='store_true', help='Resume training from existing checkpoint')
    return parser.parse_args()

args = parse_arguments()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Policy_Network(nn.Module):
    def __init__(self, input_dim,output_dim, hidden1_dim=8, hidden2_dim=8):
        super(Policy_Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.policy_mean = nn.Linear(hidden2_dim, output_dim)
        self.policy_std = nn.Linear(hidden2_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mean = self.tanh(self.policy_mean(x))
        std = F.sigmoid(self.policy_std(x))
        return mean, std

    
class Q_Network(nn.Module):
    def __init__(self, input_dim, output_dim, hidden1_dim=8, hidden2_dim=8):
        super(Q_Network, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden2_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# Replay Buffer
class ReplayBuffer:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        batch =  random.sample(self.memory, batch_size)
        return batch

    def __len__(self):
        return len(self.memory)
    
class DDPG:
    def __init__(self, env):
        self.env = env
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        self.action_limit = float(env.action_space.high[0])
        self.gamma = 0.99
        self.tau = 0.005
        self.lr = 0.001

        # Actor
        self.actor = Policy_Network(self.state_dim, self.action_dim).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-3)

        #Critic
        self.critic = Q_Network(self.state_dim + self.action_dim, 1).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        #Replay Buffer
        self.memory = ReplayBuffer(1000000)
        self.batch_size = 256
       

        self.num_params = sum(p.numel() for p in self.actor.parameters())
        print("Number of parameters: ", self.num_params,flush=True)
        self.beta = 0.1

        self.expected_delta = torch.zeros(self.num_params,device=device)
        self.expected_delta_squared = torch.zeros(self.num_params, device=device)
        self.expected_delta_delta_T = torch.zeros(self.num_params, self.num_params, device=device)

    def select_actions(self,states,is_target=False):
        if is_target:
            means, stds = self.actor_target(states)
        else:
            means, stds = self.actor(states)

        actions = []
        log_probs = []

        for mean, std in zip(means, stds):
            action, log_prob = self.sample_action(mean, std)
            actions.append(action)
            log_probs.append(log_prob)

        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)

        return actions, log_probs

        
    def sample_action(self,mean,std):
        cov_matrix = torch.diag(std.pow(2))
        eps  = 0.001
        cov_matrix = cov_matrix + eps * torch.eye(len(std),device=device)
        action_distribution = MultivariateNormal(mean, cov_matrix)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        action = self.action_limit * torch.tanh(action)
        return action, log_prob
        
        
    def perturbate(self, delta):
        flat_params = torch.cat([p.flatten() for p in self.actor.parameters()])
        flat_params = flat_params.to(device)
        delta = delta.to(device)
        flat_params_perturbed = flat_params + delta

        start_idx = 0
        for param in self.actor.parameters():
            end_idx = start_idx + param.numel()
            param.data = flat_params_perturbed[start_idx:end_idx].reshape(param.shape)
            start_idx = end_idx
        return flat_params
 
    def update(self):
        batch = self.memory.sample(self.batch_size)
        states, actions,next_states,rewards, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.float32).to(device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(device)

        # Update Critic
        self.critic_optimizer.zero_grad()

        with torch.no_grad():
            next_actions,log_probs = self.select_actions(next_states,is_target=True)
            target_q_values = self.critic_target(torch.cat([next_states, next_actions], dim=1))
            target_q_values = rewards + (1 - dones) * self.gamma * target_q_values
        
        current_q_values = self.critic(torch.cat([states, actions], dim=1))
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Policy loss is  E[delta]*Q(s,a)*1/beta + E[delta*deltaT*Q(s,a)*grad(log(pi(a|s)))] + beta* E[delta^2]
        estimated_q = 0
        loss_pi = 0

        for i in range(100):
            delta = torch.randn(self.num_params,device=device)
            self.expected_delta =  self.expected_delta +0.1*(delta-self.expected_delta)
            self.expected_delta_squared += (delta ** 2 - self.expected_delta_squared) * 0.1

            delta_delta_T = torch.outer(delta, delta)
            self.expected_delta_delta_T += (delta_delta_T - self.expected_delta_delta_T) * 0.1

            state , info = self.env.reset()
            state = torch.tensor(state,dtype=torch.float32,device=device)

            mean,std = self.actor(state)
            action,log_prob = self.sample_action(mean,std)
            q_value = self.critic(torch.cat([state.unsqueeze(0), action.unsqueeze(0)],dim=1))

            estimated_q += q_value
           
            current_loss_pi = -1*(log_prob) * (q_value.detach())
            loss_pi += current_loss_pi

        estimated_q /= 100

        self.actor_optimizer.zero_grad()
        loss_pi = loss_pi/100
        loss_pi = loss_pi*self.expected_delta_delta_T.detach()

        # loss_pi = loss_pi.mean()
        # loss_pi.backward()

        ones = torch.ones(self.expected_delta_delta_T.size(),device=device)
        loss_pi.backward(gradient=torch.Tensor(ones))
        self.actor_optimizer.step()

        noise_loss = ((estimated_q * self.expected_delta) / self.beta)*self.lr + (self.beta * self.expected_delta_squared)*self.lr

        self.perturbate(noise_loss[0])

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def train(self):
        rewards = []
        episode = -1

        if args.checkpoint:
            episode, rewards = self.load_checkpoint("Checkpoints/<checkpoint>.pth")
            print(f"Rewards: {rewards}",flush=True)
            print(f"Resuming training from episode {episode+1}",flush=True)

        for i in range(episode+1, 1000):
            state,info = self.env.reset()
            episode_reward = 0

            trajectory_length = 0

            for t in count():
                state = torch.Tensor(state).to(device)
                mean,std = self.actor(state)
                action, _ = self.sample_action(mean,std)
                action = action.detach().cpu().numpy()
                state = state.detach().cpu().numpy()
                next_state,reward,terminated,trucated,info = self.env.step(action)
                self.memory.push(state, action, next_state, reward, terminated)
                
                state = next_state
                episode_reward += reward

                print("... ",reward,flush=True)

                trajectory_length = trajectory_length + 1

                if len(self.memory) > self.batch_size:
                    self.update()

                if terminated or trajectory_length > 1000:
                    break
        
            print(f"Episode: {i}, Reward: {episode_reward}")
            rewards.append(episode_reward)

            if (i+1)%50 == 0:
                self.save_checkpoint(i,self.beta,rewards)
    
        return rewards
    
    def save_checkpoint(self,episode,beta,rewards):

        # store the replay buffer as well it is a deque
           
        checkpoint={
            'beta': beta,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'episode': episode,
            'rewards': rewards
        }

        torch.save(checkpoint, os.path.join("Humanoid", f"taylor_{episode}.pth"))

    def load_checkpoint(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        episode = checkpoint['episode']
        rewards = checkpoint['rewards']
        return episode, rewards


env = gym.make('Humanoid-v4')
actions = env.action_space.shape[0]
states = env.observation_space.shape[0]

print("Action space: ", actions,flush=True)
print("State space: ", states,flush=True)

ddpg = DDPG(env)

rewards = ddpg.train()

env.close()
folder = "Humanoid"
plt.plot(rewards)
plt.savefig(os.path.join(folder, "taylor_Humanoid_1k.png"))
