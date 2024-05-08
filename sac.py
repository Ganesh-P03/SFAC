from copy import deepcopy
import itertools
import numpy as np
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import time
import os
import sys
import matplotlib.pyplot as plt
from utils import plot_learning_curve, make_env
sys.path.insert(0,'.')

import core

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='HalfCheetah-v4')
parser.add_argument('--hid', type=int, default=8)
parser.add_argument('--l', type=int, default=2)
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--seed', '-s', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--exp_name', type=str, default='sac')
args = parser.parse_args()

class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for SAC agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

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
        #convert state to torch type double
        state = torch.as_tensor(state, dtype=torch.float32)
        # print("Convolute state:",state.shape,flush=True)
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        conv_state = conv3.view(conv3.size()[0], - 1)

        if conv_state.shape[0] != 256:
            conv_state = torch.Tensor.flatten(conv_state)

        #convert to numpy
        return conv_state.detach().numpy()


def sac(env_fn, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=1000, epochs=1000, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, lr=1e-3, alpha=0.2, batch_size=256, start_steps=256, 
        update_after=256, update_every=1, num_test_episodes=10, max_ep_len=2000, 
        logger_kwargs=dict(), save_freq=1):
    """
    Soft Actor-Critic (SAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    # logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    obs_dim = env.observation_space.shape
    act_dim = 1

    
    convolution = Convolution((obs_dim))
    conv_out_dims = convolution.calculate_conv_out_dims((obs_dim))

    print("Conv Out Dims: ",conv_out_dims,flush=True)

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(conv_out_dims, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=conv_out_dims, act_dim=act_dim, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        # conv_state = convolution.convolute(o)
        # conv_state2 = convolution.convolute(o2)
        q1 = ac.q1(o,a)
        q2 = ac.q2(o2,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        # conv_state = convolution.convolute(o)
        pi, logp_pi = ac.pi(o)
        q1_pi = ac.q1(o, pi)
        q2_pi = ac.q2(o, pi)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)

    # Set up model saving
    # logger.setup_pytorch_saver(ac)

    def update(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **q_info)

        # Freeze Q-networks so you don't waste computational effort 
        # computing gradients for them during the policy learning step.
        for p in q_params:
            p.requires_grad = False

        # Next run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        # Unfreeze Q-networks so you can optimize it at next DDPG step.
        for p in q_params:
            p.requires_grad = True

        # Record things
        # logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, deterministic=False):
        return ac.act(torch.as_tensor(o, dtype=torch.float32), 
                      deterministic)

    def test_agent():
        for j in range(num_test_episodes):
            o,i = test_env.reset()
            d, ep_ret, ep_len = False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time 
                o, r, d, _,_ = test_env.step(get_action(o, True))
                ep_ret += r
                ep_len += 1
            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o,i = env.reset()
    ep_ret, ep_len  = 0,0

    trajectories = 0

    print("State Space: ",env.observation_space,flush=True)
    print("Action Space: ",env.action_space,flush=True)

    Rewards = []

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            conv_state = convolution.convolute(o)
            a = get_action(conv_state)
            # convert a to tensor
            a = torch.tensor(a)
            a = torch.sigmoid(a)

            action = torch.distributions.Binomial(8, a).sample()
            a = action.int()

        else:
            a = env.action_space.sample()

        # if(t%1000 == 0):
        #     print("Action: ",a,flush=True)
        # Step the env
        o2, r, d, truncated,info = env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        conv_state = convolution.convolute(o)
        conv_state2 = convolution.convolute(o2)
        replay_buffer.store(conv_state, a, r, conv_state2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            trajectories += 1
            o,i = env.reset()
            print("Episode: ",trajectories," Reward: ",ep_ret," Length: ",ep_len,flush=True)
            Rewards.append(ep_ret)
            ep_ret, ep_len =  0, 0
            if(trajectories == 1000):
                checkpoint = {
                    "rewards":Rewards
                }
                torch.save(checkpoint,os.path.join("spinup",f"sac_{args.env}.pt"))
                plt.plot(Rewards)
                plt.savefig(os.path.join("spinup",f"sac_{args.env}.png"))
                break

        # Update handling
        if t >= update_after:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

if __name__ == '__main__':


    # from spinup.utils.run_utils import setup_logger_kwargs
    # logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    torch.set_num_threads(torch.get_num_threads())

    print(args.env,flush=True)

    sac(lambda : make_env(args.env), actor_critic=core.MLPActorCritic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
        gamma=args.gamma, seed=args.seed, epochs=3000)
