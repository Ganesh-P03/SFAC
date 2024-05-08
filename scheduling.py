import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import MultivariateNormal,Cauchy
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import argparse

# The max number of steps to run any trajectory
TRAJECTORY_LENGTH = 500

PERTURBATIONS = 10
GRADIENT_STEPS = 6250

# The number of samples to take for each perturbation to calculate the expected return for that perturbation
SAMPLES = 10
betas = [10,5,2.5,1.25,0.62,0.31,0.15,0.08]


#folder to save the checkpoints
CK_FOLDER = "Checkpoints"

#folder to save the training images
IMG_FOLDER = "InvertedDoublePendulum"
training_img = f"Gaussian_IDP_{PERTURBATIONS}p_{TRAJECTORY_LENGTH}.png"
env = gym.make('InvertedDoublePendulum-v4')

print("Start time: ", time.ctime(),flush=True)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Reinforcement Learning')
    parser.add_argument('-f', '--checkpoint', action='store_true', help='Resume training from existing checkpoint')
    parser.add_argument('-c', '--cauchy', action='store_true', help='Use Cauchy perturbations')
    return parser.parse_args()

args = parse_arguments()

if args.cauchy:
    print("Using Cauchy perturbations",flush=True)
else:
    print("Using Gaussian perturbations",flush=True)

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device,flush=True)

if device.type == "cuda":
    print("Number of GPUs available:", torch.cuda.device_count(),flush=True)
    print("Current GPU:", torch.cuda.current_device(),flush=True)
    print("GPU Name:", torch.cuda.get_device_name(torch.cuda.current_device()),flush=True)


class Policy_Network(nn.Module):
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
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
    
class REINFORCE():
    def __init__(self, env, policy_network, alpha, gamma, max_episode, device):
        self.env = env
        self.policy_network = policy_network.to(device)
        self.alpha = alpha
        self.gamma = gamma
        self.num_params = sum(p.numel() for p in policy_network.parameters())
        print("number of parameters: ", self.num_params,flush=True)
        self.max_episode = max_episode
        self.action_limit = float(env.action_space.high[0])
        print("Action limit: ", self.action_limit,flush=True)
       
        self.device = device

    def train(self,idx=-1):
        global_returns = []

        if(idx == len(betas)-1):
            return global_returns
        
        for i_beta,beta in enumerate(betas[idx+1:]):
            i_beta = (idx+1)+i_beta

            for k in range(0,GRADIENT_STEPS):
                grad_returns = torch.zeros(self.num_params,device=self.device)                
                for i_episode in range(max(self.max_episode,1)):
                    delta = None

                    if args.cauchy:
                        delta = Cauchy(torch.zeros(self.num_params,device=self.device),torch.ones(self.num_params,device=self.device)).sample() * beta
                    else: 
                        delta = torch.randn(self.num_params,device=self.device)*beta

                    if self.max_episode == 0:
                        delta = delta-delta 
                    previous_flat_params = self.perturbate(delta)

                    expected_return = 0
                
                    for j_episode in range(SAMPLES):
                        tau = self.sample_trajectory() 
                        expected_return += np.sum([self.gamma**i * tau[i][2] for i in range(0, len(tau))]) 
                    
                    expected_return = expected_return/SAMPLES
                    
                    grad_return = None

                    if args.cauchy:
                        grad_return = ((delta*(self.num_params+1)) * expected_return)/(beta*(1+torch.inner(delta,delta)))
                    else:
                        grad_return = (expected_return*delta)/beta

                    grad_returns += grad_return
                    self.restore(previous_flat_params)

                grad_returns = grad_returns/max(self.max_episode,1)

                expected_return = 0
                for j_episode in range(SAMPLES):
                    tau = self.sample_trajectory() 
                    expected_return += np.sum([self.gamma**i * tau[i][2] for i in range(0, len(tau))]) 
                expected_return = expected_return/SAMPLES

                print("expected_returns #", k,"beta:",beta, expected_return)
                self.perturbate(grad_returns*self.alpha)

            
            self.save_checkpoint(beta,i_beta,global_returns)
            
        return global_returns
    
    def save_checkpoint(self, beta, i_beta, global_returns):
        checkpoint = {
            'beta': beta,
            'i_beta': i_beta,
            'model_state_dict': self.policy_network.state_dict(),
            'global_returns': global_returns,    
        }

        torch.save(checkpoint, os.path.join(CK_FOLDER, f'{i_beta}_{beta}.pth'))
        print(f"Checkpoint {i_beta} saved",flush=True)
    
    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.policy_network.load_state_dict(checkpoint['model_state_dict'])
        i_beta = checkpoint['i_beta']
        global_returns = checkpoint['global_returns']
        return i_beta, global_returns

    def perturbate(self, delta):
        flat_params = torch.cat([p.flatten() for p in self.policy_network.parameters()])
        flat_params = flat_params.to(self.device)
        delta = delta.to(self.device)
        flat_params_perturbed = flat_params + delta

        start_idx = 0
        for param in self.policy_network.parameters():
            end_idx = start_idx + param.numel()
            param.data = flat_params_perturbed[start_idx:end_idx].reshape(param.shape)
            start_idx = end_idx
        return flat_params
    
    def restore(self, flat_params):
        # Reset the model parameters
        start_idx = 0
        for param in self.policy_network.parameters():
            end_idx = start_idx + param.numel()
            param.data = flat_params[start_idx:end_idx].reshape(param.shape)
            start_idx = end_idx

    def sample_trajectory(self):
        tau = []
        observation, info = self.env.reset()
        for t in range(TRAJECTORY_LENGTH ):
            observation = torch.tensor(observation, dtype=torch.float32).to(self.device)
            action, log_prob = self.sample_action(observation)
            observation, reward, terminated, truncated, info = self.env.step(action)
            tau.append((observation, action, reward, log_prob))
              
            if terminated:
                break
        return tau

    def sample_action(self, observation):
        action_mean, action_std = self.policy_network(observation)
        cov_matrix = torch.diag(action_std.pow(2))
        eps  = 0.001
        cov_matrix = cov_matrix + eps * torch.eye(len(action_std), device=self.device)
        action_distribution = MultivariateNormal(action_mean, cov_matrix)
        action = action_distribution.sample()
        log_prob = action_distribution.log_prob(action)
        action = self.action_limit*torch.tanh(action)
        action = action.detach().cpu().numpy()
        return action, log_prob
    
# Train

actions = env.action_space.shape[0]
states = env.observation_space.shape[0]

print("Action space: ", actions,flush=True)
print("State space: ", states,flush=True)

policy_network = Policy_Network(states, 8,8, actions).to(device)
gamma = 1
# default learning rate for adam optimizer 
lr = 0.001

reinforce = REINFORCE(env, policy_network, lr,gamma,PERTURBATIONS, device)
avg_episodic_rewards = []

if not os.path.exists(IMG_FOLDER):
    os.makedirs(IMG_FOLDER)

if args.checkpoint:
    if not os.path.exists(CK_FOLDER):
        os.makedirs(CK_FOLDER)

train_returns = None

if os.path.exists(CK_FOLDER):
    if(len(os.listdir(CK_FOLDER)) == 0) or not args.checkpoint:
        train_returns = reinforce.train()
    else:
        latest_checkpoint = max([os.path.join(CK_FOLDER, f) for f in os.listdir(CK_FOLDER)], key=os.path.getctime)
        print("Loading checkpoint", latest_checkpoint,flush=True)
        i_beta,global_returns = reinforce.load_checkpoint(latest_checkpoint)
        train_returns = global_returns
        new_returns  = reinforce.train(i_beta)
        train_returns.extend(new_returns)
else:
    # Start training from scratch 
    train_returns = reinforce.train()

print("Training complete",flush=True)
text_string = f'Trajectory length: {TRAJECTORY_LENGTH}\n Perturbations: {PERTURBATIONS}\n lr: {lr}'
# Plot
plt.plot(train_returns)
plt.xlabel('Episode')
plt.ylabel('Average Reward')

if args.cauchy:
    plt.title(f"Cauchy Schedule for IDP {betas}")
else:
    plt.title(f"Gaussian Schedule for IDP {betas}")

plt.text(0.95, 0.95, text_string, horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes)
plt.savefig(os.path.join(IMG_FOLDER,training_img))

print("End time: ", time.ctime(),flush=True)
print("Duration: ", time.process_time()/3600, "hours",flush=True)