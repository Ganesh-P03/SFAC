import gymnasium as gym
import numpy as np
import torch

from stable_baselines3.ddpg.policies import MlpPolicy
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
from stable_baselines3 import DDPG,SAC
import matplotlib.pyplot as plt


env = gym.make('Hopper-v4')

# the noise objects for DDPG
n_actions = env.action_space.shape[-1]
action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=float(0.5) * np.ones(n_actions))

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[8, 8])
# model = DDPG(MlpPolicy, env, verbose=1, action_noise=action_noise)
model = SAC('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
info = model.learn(total_timesteps=1000)
print("info: ",info.__dict__)

total_rewards = []


for i in range(500):
    obs,info = env.reset()
    t_reward =0 
    timesteps = 0
    while True:
        action, _states = model.predict(obs)
        obs, rewards, terminated,truncated,info = env.step(action)
        t_reward += rewards
        # print("... ",rewards,flush=True)
        timesteps += 1
        if terminated or timesteps > 1000:
            break
    print("Episode: ", i, "Reward: ", t_reward)
    total_rewards.append(t_reward)

print("Mean reward: ", np.mean(total_rewards))

plt.plot(total_rewards)
plt.savefig("SB_SAC_Hopper_1k.png")
    