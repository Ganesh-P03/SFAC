#env: CrazyClimberNoFrameskip-v4
# to run keep --env CrazyClimberNoFrameskip-v4

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
import numpy as np
from utils import plot_learning_curve, make_env

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reinforcement Learning')
    parser.add_argument('-c', '--checkpoint', action='store_true', help='Resume training from existing checkpoint')
    return parser.parse_args()

args = parse_arguments()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Convolution(nn.Module):
    def __init__(self, input_dim):
        super(Convolution, self).__init__()
        print("Input Dim: ",input_dim,flush=True)
        self.conv1 = nn.Conv2d(input_dim[0], 16, kernel_size = 8, stride = 4)
        self.conv2 = nn.Conv2d(16, 8, kernel_size = 4, stride = 2)
        self.conv3 = nn.Conv2d(8, 4, kernel_size = 4, stride = 1)
        conv_out_dims = self.calculate_conv_out_dims(input_dim)

        print("Conv Out Dims: ",conv_out_dims,flush=True)
    
    def calculate_conv_out_dims(self, input_dims):
        """
            After our convolution layers are done we need to know the
            input dimensions so that we can pass it to our fully-connected layers
            This is an aux function to calculate the input dims by just passing a tensor of zeros
            sequentially and taking the shape afterwards for use in our first fc layer.
        """
        state = torch.zeros(1, *input_dims)
        print("State shape: ",state.shape,flush=True)
        dim = self.conv1(state)
        print("Conv1 shape: ",dim.shape,flush=True)
        dim = self.conv2(dim)
        print("Conv2 shape: ",dim.shape,flush=True)
        dim = self.conv3(dim)
        print("Conv3 shape: ",dim.shape,flush=True)
        return int(np.prod(dim.size()))
    
    def convolute(self, state):
        # print("convolute_state: ",state,flush=True)
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], - 1)

        if conv_state.shape[0] != 256:
            conv_state = torch.Tensor.flatten(conv_state)

        return conv_state

              
class Policy_Network(nn.Module):
    def __init__(self, input_dim,output_dim, hidden1_dim=6, hidden2_dim=6):
        super(Policy_Network, self).__init__()
        self.policy_convolution = Convolution(input_dim)
        conv_out_dims = self.policy_convolution.calculate_conv_out_dims(input_dim)
        self.fc1 = nn.Linear(conv_out_dims, hidden1_dim)
        # self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.policy_mean = nn.Linear(hidden1_dim, output_dim)
        # self.policy_std = nn.Linear(hidden2_dim, output_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, state):
        conv_state = self.policy_convolution.convolute(state)
        x = self.relu(self.fc1(conv_state))
        # x = self.relu(self.fc2(x))
        mean = F.sigmoid(self.policy_mean(x))
        # std = F.sigmoid(self.policy_std(x))
        return mean
    
    
 
class Q_Network(nn.Module):
    def __init__(self, state_dim,action_dim, output_dim, hidden1_dim=6, hidden2_dim=6):
        super(Q_Network, self).__init__()

        self.q_convolution = Convolution(state_dim)
        conv_out_dims = self.q_convolution.calculate_conv_out_dims(state_dim)

        input_dim = conv_out_dims + action_dim
        self.fc1 = nn.Linear(input_dim, hidden1_dim)
        # self.fc2 = nn.Linear(hidden1_dim, hidden2_dim)
        self.fc3 = nn.Linear(hidden1_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, state , action):
        conv_state = self.q_convolution.convolute(state)
        # print("Conv State: ",conv_state.shape,flush=True)
        if action.shape[0]!=256:
            action = torch.Tensor.flatten(action)
            x  = torch.cat([conv_state, action], dim=0)
        else:
            x  = torch.cat([conv_state, action], dim=1)

        # print("Action: ",action.shape,flush=True)
        # print(conv_state)
        # print(action)
        x = self.relu(self.fc1(x))
        # x = self.relu(self.fc2(x))
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
        self.state_dim = env.observation_space.shape
        #if discrete action space =1

        if isinstance(env.action_space, gym.spaces.Discrete):
            self.action_dim = 1
        else:
            self.action_dim = 1
        
        # self.action_limit = float(env.action_space.high[0])
        self.gamma = 0.5
        self.tau_actor = 0.001
        self.tau_critic = 0.003
        self.lr = 0.01

        # Actor
        self.actor = Policy_Network((self.state_dim), self.action_dim).to(device)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-2)

        #Critic
        print("State Dim: ",self.state_dim,flush=True)
        print("Action Dim: ",self.action_dim,flush=True)
        # flatten the state
        # state_dim = self.state_dim[0]*self.state_dim[1]*self.state_dim[2]
        self.critic = Q_Network((self.state_dim) , self.action_dim,1).to(device)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-1)

        #Replay Buffer
        self.memory = ReplayBuffer(256)
        self.batch_size = 256
       

        self.num_params = sum(p.numel() for p in self.actor.parameters())
        print("Number of parameters: ", self.num_params,flush=True)
        self.beta = 0.1
        self.alpha = 1.01

        self.expected_delta = torch.zeros(self.num_params,device=device)
        self.expected_delta_squared = torch.zeros(self.num_params, device=device)
        self.expected_delta_delta_T = torch.zeros(self.num_params, self.num_params, device=device)

    def select_actions(self,states,is_target=False):
        if is_target:
            means = self.actor_target(states)
        else:
            means = self.actor(states)

        actions = []
        log_probs = []

        for mean in means:
            action, log_prob = self.sample_action(mean)
            actions.append(action)
            log_probs.append(log_prob)

        actions = torch.stack(actions)
        log_probs = torch.stack(log_probs)

        return actions, log_probs

        
    def sample_action(self,mean):
        action = torch.distributions.Binomial(8, mean).sample()
        log_prob = torch.distributions.Binomial(8, mean).log_prob(action)

       
        action = action.int()


        # print("Device for action:",action.device,flush=True)

        epsilon = 0.1

        if random.random() < epsilon:
            action = torch.distributions.Categorical(torch.tensor([1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9,1/9])).sample()
            # log_prob = torch.distributions.Categorical(torch.tensor([0.33,0.33,0.33])).log_prob(action)

            # print("Epsilon Action:",action.device,flush=True)
            # cuda tensor array
            action = torch.tensor([action.item()],device=device)
            log_prob = torch.distributions.Binomial(8, mean).log_prob(action)
            # print("Epsilon Log Prob: ",log_prob,flush=True)
            action = action.int()
            return action, log_prob
        
        # print("Log Prob: ",log_prob,flush=True)
            
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
            target_q_values = self.critic_target(next_states, next_actions)
            # print("Target Q Values: ",target_q_values.shape,flush=True)
            # print("Rewards: ",rewards.shape,flush=True)
            target_q_values = rewards + (1 - dones) * self.gamma * (target_q_values - self.alpha * log_probs)
        
        current_q_values = self.critic(states,actions)
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        critic_loss.backward()
        self.critic_optimizer.step()

        # Policy loss is  E[delta]*Q(s,a)*1/beta + E[delta*deltaT*Q(s,a)*grad(log(pi(a|s)))] + beta* E[delta^2]
        estimated_q = 0
        loss_pi = 0

        for i in range(40):
            delta = torch.randn(self.num_params,device=device)
            self.expected_delta =  self.expected_delta +0.01*(delta-self.expected_delta)
            self.expected_delta_squared += (delta ** 2 - self.expected_delta_squared) * 0.01

            delta_delta_T = torch.outer(delta, delta)
            self.expected_delta_delta_T += (delta_delta_T - self.expected_delta_delta_T) * 0.01

            state , info = self.env.reset()
            state = torch.tensor(state,dtype=torch.float32,device=device)

            mean = self.actor(state)
            action,log_prob = self.sample_action(mean)
            q_value = self.critic(state.unsqueeze(0), action.unsqueeze(0))

            estimated_q += q_value
           
            current_loss_pi = -1*(log_prob) * (q_value.detach() - self.alpha * log_prob.detach() - 1)
            loss_pi += current_loss_pi

        estimated_q /= 40

        self.actor_optimizer.zero_grad()
        loss_pi = loss_pi/40
        loss_pi = loss_pi*self.expected_delta_delta_T.detach()

        # loss_pi = loss_pi.mean()
        # loss_pi.backward()

        ones = torch.ones(self.expected_delta_delta_T.size(),device=device)
        loss_pi.backward(gradient=torch.Tensor(ones))
        self.actor_optimizer.step()

        noise_loss = ((estimated_q * self.expected_delta) / self.beta)*self.lr + (self.beta * self.expected_delta_squared)*self.lr

        self.perturbate(noise_loss[0])

        for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_param.data.copy_(self.tau_actor * param.data + (1 - self.tau_actor) * target_param.data)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau_critic * param.data + (1 - self.tau_critic) * target_param.data)

    def train(self):
        rewards = []
        episode = -1

        if args.checkpoint:
            episode, rewards = self.load_checkpoint("Checkpoints/taylor_9.pth")
            print(f"Rewards: {rewards}",flush=True)
            print(f"Resuming training from episode {episode+1}",flush=True)

        for i in range(episode+1, 10000):
            state,info = self.env.reset()
            episode_reward = 0

            trajectory_length = 0
            update_every = 0

            for t in count():
                state = torch.Tensor(state).to(device)
                mean = self.actor(state)
                action, _ = self.sample_action(mean)
                action = action.detach().cpu().numpy()
                state = state.detach().cpu().numpy()
                next_state,reward,terminated,truncated,info = self.env.step(action[0])
                self.memory.push(state, action, next_state, reward, terminated)
                
                state = next_state
                episode_reward += reward

                #print("... ",reward,flush=True)

                trajectory_length = trajectory_length + 1
                update_every = update_every + 1

                if len(self.memory) > self.batch_size:
                    # if(update_every % 20 == 0):
                    self.update()

                if terminated or truncated or trajectory_length > 2000:
                    update_every = 0
                    break
        
            print(f"Episode: {i}, Reward: {episode_reward}", flush=True)
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

        torch.save(checkpoint, os.path.join("Atari", f"taylor_{episode}.pth"))

    def load_checkpoint(self,checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        episode = checkpoint['episode']
        rewards = checkpoint['rewards']
        return episode, rewards


env = make_env('CrazyClimberNoFrameskip-v4')
actions = env.action_space.n
states = env.observation_space.shape[0]

print("Action space: ", actions,flush=True)
print("State space: ", states,flush=True)

ddpg = DDPG(env)

rewards = ddpg.train()

env.close()

plt.plot(rewards)
plt.savefig("taylor_Acrobat_1.png")
#acrobat kept 100

    

    
