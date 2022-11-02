# Modified code based on the OpenAI Spinningup repository: https://github.com/openai/spinningup/blob/master/spinup/algos/pytorch/td3/td3.py

from copy import deepcopy
import itertools
from typing import Dict
import numpy as np
import torch
from torch.optim import Adam
from torch.optim import lr_scheduler
import gym
import time
import algos.td3_core as core
# from spinup.utils.logx import EpochLogger
import os
import h5py
from utils.training import printProgressBar
from utils.training import get_data_loader
from utils.encoding import encode_obs
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.utils.obs_utils as ObsUtils

from utils.logger import MyLogger

import imageio


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer for TD3 agents.
    """

    def __init__(self, obs_dim, act_dim, size, expert_data_dict=None, encoder=None, obs_normalization_stats=None, env=None, demo_size=0, reward_correction=None):

        print("Creating the replay buffer...")
        self.demo_size = demo_size

        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

        if expert_data_dict is not None:

            for key, expert_data in expert_data_dict.items():

                actions_batch = expert_data['actions'].numpy()
                dones_batch = expert_data['dones'].numpy()
                rewards_batch = expert_data['rewards'].numpy()

                actions_batch = np.squeeze(actions_batch)
                dones_batch = np.squeeze(dones_batch)
                rewards_batch = np.squeeze(rewards_batch)

                if reward_correction is not None:
                    rewards_batch = rewards_batch + reward_correction

                if encoder is not None:
                    print("Encoding demonstration data observations...")

                    input_batch = encoder.process_batch_for_training(expert_data)

                    batch_prep = TensorUtils.to_device(TensorUtils.to_float(input_batch), device='cuda')
                    latent_obs = encoder.nets['policy'].nets['encoder'].forward(input=batch_prep['obs'])['mean']

                    input_batch = dict()
                    input_batch["next_obs"] = {k: expert_data["next_obs"][k][:, 0, :] for k in expert_data["next_obs"]}

                    batch_prep = TensorUtils.to_device(TensorUtils.to_float(input_batch), device='cuda')
                    latent_next_obs = encoder.nets['policy'].nets['encoder'].forward(input=batch_prep['next_obs'])['mean']

                    obs_batch = latent_obs.cpu().detach().numpy()
                    next_obs_batch = latent_next_obs.cpu().detach().numpy()
                else:
                    obs_lst = []
                    next_obs_lst = []
                    for i in range(expert_data["actions"].shape[0]):
                        sample_lst = dict()
                        sample_lst['obs'] = {key.replace('object', 'object-state') : expert_data['obs'][key][i, 0, :] for key in expert_data['obs']}
                        sample_lst['next_obs'] = {key.replace('object', 'object-state') : expert_data['next_obs'][key][i, 0, :] for key in expert_data['next_obs']}

                        obs_lst.append(env._flatten_obs(sample_lst['obs']))
                        next_obs_lst.append(env._flatten_obs(sample_lst['next_obs']))

                    obs_batch = np.array(obs_lst)
                    next_obs_batch = np.array(next_obs_lst)

                # if (key == 'success') or ("success" not in expert_data_dict):
                if (key == 'success'):
                    self.store_demonstration(obs_batch=obs_batch, 
                                        act_batch=actions_batch,
                                        rew_batch=rewards_batch,
                                        next_obs_batch=next_obs_batch,
                                        done_batch=dones_batch)
                else:
                    self.store_batch(obs_batch=obs_batch, 
                                        act_batch=actions_batch,
                                        rew_batch=rewards_batch,
                                        next_obs_batch=next_obs_batch,
                                        done_batch=dones_batch)

            print("Demonstration loading is done")

        print("Replay buffer is created")


    def store(self, obs, act, rew, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        # self.ptr = (self.ptr+1) % self.max_size

        if self.ptr+1 == self.max_size:
            self.ptr = self.demo_size
        else:
            self.ptr += 1

        self.size = min(self.size+1, self.max_size)

    def store_demonstration(self, obs_batch, act_batch, rew_batch, next_obs_batch, done_batch):
        batch_size = obs_batch.shape[0]

        assert (batch_size <= self.max_size), f"The demonstrations (size: {batch_size}) cannot be fit into the replay buffer (size: {self.max_size})"

        self.obs_buf[0:batch_size] = obs_batch
        self.obs2_buf[0:batch_size] = next_obs_batch
        self.act_buf[0:batch_size] = act_batch
        self.rew_buf[0:batch_size] = rew_batch
        self.done_buf[0:batch_size] = done_batch
        self.ptr = batch_size
        self.size = batch_size
        self.demo_size = batch_size

    def store_batch(self, obs_batch, act_batch, rew_batch, next_obs_batch, done_batch):
        batch_size = obs_batch.shape[0]

        for i in range(batch_size):
            self.store(obs_batch[i,], act_batch[i,], rew_batch[i,], next_obs_batch[i,], done_batch[i,])
        

    def sample_batch(self, batch_size=400):
        if self.demo_size > 0:
            num_of_demos = np.random.randint(low=0, high=batch_size)
            idxs = np.random.randint(0, self.demo_size, size=num_of_demos)
            idxs = np.concatenate((idxs, np.random.randint(0, self.size, size=batch_size-num_of_demos)))
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
        # idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}

    def set_demo_size(self, demo_size):
        self.demo_size = demo_size



def td3(env, actor_critic=core.MLPActorCritic, ac_kwargs=dict(), seed=0, 
        steps_per_epoch=4000, epochs=100, replay_size=int(1e6), gamma=0.99, 
        polyak=0.995, pi_lr=1e-3, q_lr=1e-3, batch_size=100, start_steps=10000, 
        update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2, 
        noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000, 
        logger_kwargs=dict(), save_freq=1, pretrain_on_demonstration=False,
        pretrain_steps=10, encoder=None, force_scale=None, expert_data_dict=None, obs_normalization_stats=None, 
        expert_data_path=None, fix_scenario=False, reward_correction=None, success_boost=None):
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3)
    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.
        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            these should return:
            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``pi``       (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================
        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to TD3.
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
        pi_lr (float): Learning rate for policy.
        q_lr (float): Learning rate for Q-networks.
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
        act_noise (float): Stddev for Gaussian exploration noise added to 
            policy at training time. (At test time, no noise is added.)
        target_noise (float): Stddev for smoothing noise added to target 
            policy.
        noise_clip (float): Limit for absolute value of target policy 
            smoothing noise.
        policy_delay (int): Policy will only be updated once every 
            policy_delay times for each update of the Q-networks.
        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.
        max_ep_len (int): Maximum length of trajectory / episode / rollout.
        logger_kwargs (dict): Keyword args for EpochLogger.
        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.
    """

    # logger = EpochLogger(**logger_kwargs)
    # logger.save_config(locals())

    mylogger = MyLogger()
    

    # mylogger = dict()
    # mylogger['losses'] = dict()
    # mylogger['losses']['LossQ'] = []
    # mylogger['losses']['LossPi'] = []

    # mylogger['EpRet'] = []
    # mylogger['EpLen'] = []
    # mylogger['TestEpRet'] = []
    # mylogger['TestEpLen'] = []

    torch.manual_seed(seed)
    np.random.seed(seed)

    o = env.reset()

    if encoder is not None:
        o = encode_obs(encoder=encoder, obs=o, obs_normalization_stats=obs_normalization_stats)

    print("Creating experience collector and test environment")

    # env, test_env = env_fn(), env_fn()
    # env, test_env = env_fn, env_fn

    if encoder is not None:
        obs_dim = encoder.algo_config.vae["latent_dim"]
    else:
        obs_dim = env.observation_space.shape[0]
    
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(obs_dim, env.action_space, **ac_kwargs)
    # ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    # replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size)
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, 
        act_dim=act_dim, 
        size=replay_size, 
        expert_data_dict=expert_data_dict,
        encoder=encoder,
        obs_normalization_stats=obs_normalization_stats,
        env=env,
        reward_correction=reward_correction)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    # logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)
    print("\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n"%var_counts)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o,a)
        q2 = ac.q2(o,a)

        # Bellman backup for Q functions
        with torch.no_grad():
            pi_targ = ac_targ.pi(o2)

            # Target policy smoothing
            epsilon = torch.randn_like(pi_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = pi_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2)
            q2_pi_targ = ac_targ.q2(o2, a2)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * q_pi_targ
            # backup = r + gamma * q_pi_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.detach().numpy(),
                         Q2Vals=q2.detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 pi loss
    def compute_loss_pi(data):
        o = data['obs']
        q1_pi = ac.q1(o, ac.pi(o))
        return -q1_pi.mean()

    # Set up optimizers for policy and q-function
    pi_optimizer = Adam(ac.pi.parameters(), lr=pi_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    # q_scheduler = lr_scheduler.ExponentialLR(q_optimizer, gamma=0.8)
    # pi_scheduler = lr_scheduler.ExponentialLR(pi_optimizer, gamma=0.99)
    q_scheduler = lr_scheduler.ExponentialLR(q_optimizer, gamma=0.99)
    pi_scheduler = lr_scheduler.ExponentialLR(pi_optimizer, gamma=0.99)

    # Set up model saving
    # logger.setup_pytorch_saver(ac)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Record things
        # logger.store(LossQ=loss_q.item(), **loss_info)
        mylogger.store(LossQ=loss_q.item())
        # mylogger["losses"]["LossQ"].append(loss_q.item())

        # Possibly update pi and target networks
        if timer % policy_delay == 0:

            # Freeze Q-networks so you don't waste computational effort 
            # computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for pi.
            pi_optimizer.zero_grad()
            loss_pi = compute_loss_pi(data)
            loss_pi.backward()
            pi_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Record things
            # logger.store(LossPi=loss_pi.item())
            mylogger.store(LossPi=loss_pi.item())
            # mylogger["losses"]["LossPi"].append(loss_pi.item())

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # NB: We use an in-place operations "mul_", "add_" to update target
                    # params, as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, noise_scale):
        a = ac.act(torch.as_tensor(o, dtype=torch.float32))
        a += noise_scale * np.random.randn(act_dim)
        return np.clip(a, -act_limit, act_limit)

    # set download folder and make it
    download_folder = "/tmp/robomimic_ds_example"
    os.makedirs(download_folder, exist_ok=True)

    # prepare to write playback trajectories to video
    video_path = os.path.join(download_folder, "playback.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    def test_agent():
        for j in range(num_test_episodes):
            # o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0

            o, d, ep_ret, ep_len = env.reset(), False, 0, 0

            if fix_scenario:
                if hasattr(env.unwrapped.sim, "set_state_from_flattened"):
                    env.unwrapped.sim.set_state_from_flattened(initial_state_dict["states"])
                    env.unwrapped.sim.forward()
                    obs_dict = env.unwrapped._get_observations()
                    o = env._flatten_obs(obs_dict)
                else:
                    raise NotImplementedError

            # o, d, ep_ret, ep_len = env.reset(), False, 0, 0
            o, d, ep_ret, ep_len = o, False, 0, 0

            if encoder is not None:
                o = encode_obs(encoder=encoder, obs=o, obs_normalization_stats=obs_normalization_stats)

            if env.unwrapped.has_renderer:
                env.render()
            elif env.unwrapped.has_offscreen_renderer:
                # video_img = env.render(mode="rgb_array", height=512, width=512, camera_name="agentview")
                video_img = env.sim.render(height=512, width=512, camera_name='frontview')[::-1]
                video_writer.append_data(video_img)

            # while not(d or (ep_len == max_ep_len)):
            while not((ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                # o, r, d, _ = test_env.step(get_action(o, 0))
                o, r, d, _ = env.step(get_action(o, 0))

                d = int(env._check_success())

                if d:
                    r += success_boost

                if reward_correction is not None:
                    r += reward_correction

                ep_ret += r
                ep_len += 1

                if encoder is not None:
                    o = encode_obs(encoder=encoder, obs=o, obs_normalization_stats=obs_normalization_stats)

                if env.unwrapped.has_renderer:
                    env.render()
                elif env.unwrapped.has_offscreen_renderer:
                    # video_img = env.render(mode="rgb_array", height=512, width=512, camera_name="agentview")
                    video_img = env.sim.render(height=512, width=512, camera_name='frontview')[::-1]
                    video_writer.append_data(video_img)

            # logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            mylogger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            # mylogger['TestEpRet'].append(ep_ret)
            # mylogger['TestEpLen'].append(ep_len)
            print(f"Episode return after test: {ep_ret}         Episode length: {ep_len}         End success: {d}")

    
    if pretrain_on_demonstration:
        print("Pretraining on the demonstration data")
        for i in range(pretrain_steps):
            # batch = replay_buffer.sample_batch(batch_size)
            # update(data=batch, timer=0)
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)

            if i % 100 == 0:
                print(f"Pretrain steps: {i}/{pretrain_steps}")
                q_scheduler.step()
                pi_scheduler.step()
        
        # mylogger.plot(['LossQ', 'LossPi'])

    print("Prepare for interaction with the environment")

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs

    demo_counter = 0
    demo_len = -1
    if fix_scenario:
        demo_key = "demo_0"
        with h5py.File(expert_data_path["success"], "r") as f:
            demo_actions = f["data/{}/actions".format(demo_key)][:]

        demo_len = demo_actions.shape[0]

        replay_buffer.set_demo_size(update_after)


    start_time = time.time()

    o, ep_ret, ep_len = env.reset(), 0, 0

    if fix_scenario:
        with h5py.File(expert_data_path["success"], "r") as f:
            demo_key = "demo_0"
            init_state = f["data/{}/states".format(demo_key)][0]
            model_xml = f["data/{}".format(demo_key)].attrs["model_file"]
            initial_state_dict = dict(states=init_state, model=model_xml)

        if hasattr(env.unwrapped.sim, "set_state_from_flattened"):
            env.unwrapped.sim.set_state_from_flattened(initial_state_dict["states"])
            env.unwrapped.sim.forward()
            obs_dict = env.unwrapped._get_observations()
            o = env._flatten_obs(obs_dict)
        else:
            raise NotImplementedError

    # o, ep_ret, ep_len = env.reset(), 0, 0
    o, d, ep_ret, ep_len = o, False, 0, 0

    if encoder is not None:
        o = encode_obs(encoder=encoder, obs=o, obs_normalization_stats=obs_normalization_stats)

    if env.unwrapped.has_renderer:
        env.render()

    reward_history = [ep_ret]
    signal_history = [o]
    action_history = []

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):

        env_step_time_start = time.time()
        
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy (with some noise, via act_noise). 
        if t > start_steps:
            a = get_action(o, act_noise)
        elif fix_scenario:
            a = demo_actions[demo_counter, :]
            a += act_noise * np.random.randn(act_dim)
            np.clip(a, -act_limit, act_limit)
            demo_counter += 1
        else:
            a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(a)

        d = int(env._check_success())

        if d:
            r += success_boost

        if reward_correction is not None:
                r += reward_correction

        ep_ret += r
        ep_len += 1

        mylogger.store(EnvStepDuration=time.time()-env_step_time_start)

        signal_history.append(o2)
        # reward_history.append(ep_ret)
        action_history.append(a)

        if encoder is not None:
            o2 = encode_obs(encoder=encoder, obs=o2, obs_normalization_stats=obs_normalization_stats)

        if env.unwrapped.has_renderer:
            env.render()

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # if d:
            # r += 100
            # ep_ret += 100

        reward_history.append(ep_ret)


        # store_start = time.time()

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d)

        # mylogger.store(StoreDuration=time.time()-store_start)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        # if d or (ep_len == max_ep_len):
        # if d or (ep_len == max_ep_len) or (demo_counter == demo_len):
        if (ep_len == max_ep_len) or (demo_counter == demo_len):
            demo_counter = 0
            # logger.store(EpRet=ep_ret, EpLen=ep_len)
            mylogger.store(EpRet=ep_ret, EpLen=ep_len)
            # mylogger['EpRet'].append(ep_ret)
            # mylogger['EpLen'].append(ep_len)
            print(f"Episode return after interaction: {ep_ret}         Episode length: {ep_len}         End success: {d}")

            o, ep_ret, ep_len = env.reset(), 0, 0

            if fix_scenario:
                if hasattr(env.unwrapped.sim, "set_state_from_flattened"):
                    env.unwrapped.sim.set_state_from_flattened(initial_state_dict["states"])
                    env.unwrapped.sim.forward()
                    obs_dict = env.unwrapped._get_observations()
                    o = env._flatten_obs(obs_dict)
                else:
                    raise NotImplementedError

            # o, ep_ret, ep_len = env.reset(), 0, 0
            o, d, ep_ret, ep_len = o, False, 0, 0

            reward_history = [ep_ret]
            signal_history = [o]
            action_history = []

            if encoder is not None:
                o = encode_obs(encoder=encoder, obs=o, obs_normalization_stats=obs_normalization_stats)

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(batch_size)
                update(data=batch, timer=j)

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            print(f"The size of data in the replay buffer: {replay_buffer.size}/{replay_buffer.max_size}")
            epoch = (t+1) // steps_per_epoch

            print(f"Epoch {epoch}")

            if t > start_steps:
                q_scheduler.step()
                pi_scheduler.step()

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                ckpt_folder = '/home/rvarga/Data/Delft/thesis/implementation/robosuite/custom/ckpt'
                torch.save(ac.state_dict(), os.path.join(ckpt_folder, f"epoch{epoch}" + ".pth"))
                # logger.save_state({'env': env}, None)

            print("Testing the agent")

            # Test the performance of the deterministic version of the agent.
            test_agent()

            o, ep_ret, ep_len = env.reset(), 0, 0

            if fix_scenario:
                if hasattr(env.unwrapped.sim, "set_state_from_flattened"):
                    env.unwrapped.sim.set_state_from_flattened(initial_state_dict["states"])
                    env.unwrapped.sim.forward()
                    obs_dict = env.unwrapped._get_observations()
                    o = env._flatten_obs(obs_dict)
                else:
                    raise NotImplementedError

            # o, ep_ret, ep_len = env.reset(), 0, 0
            o, d, ep_ret, ep_len = o, False, 0, 0

            if encoder is not None:
                o = encode_obs(encoder=encoder, obs=o, obs_normalization_stats=obs_normalization_stats)

            # Log info about epoch
            # logger.log_tabular('Epoch', epoch)
            # logger.log_tabular('EpRet', with_min_and_max=True)
            # logger.log_tabular('TestEpRet', with_min_and_max=True)
            # logger.log_tabular('EpLen', average_only=True)
            # logger.log_tabular('TestEpLen', average_only=True)
            # logger.log_tabular('TotalEnvInteracts', t)
            # logger.log_tabular('Q1Vals', with_min_and_max=True)
            # logger.log_tabular('Q2Vals', with_min_and_max=True)
            # logger.log_tabular('LossPi', average_only=True)
            # logger.log_tabular('LossQ', average_only=True)
            # logger.log_tabular('Time', time.time()-start_time)
            # logger.dump_tabular()

    if env.unwrapped.has_offscreen_renderer:
            # done writing video
            video_writer.close()

# if __name__ == '__main__':
#     import argparse
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--env', type=str, default='HalfCheetah-v2')
#     parser.add_argument('--hid', type=int, default=256)
#     parser.add_argument('--l', type=int, default=2)
#     parser.add_argument('--gamma', type=float, default=0.99)
#     parser.add_argument('--seed', '-s', type=int, default=0)
#     parser.add_argument('--epochs', type=int, default=50)
#     parser.add_argument('--exp_name', type=str, default='td3')
#     args = parser.parse_args()

#     from spinup.utils.run_utils import setup_logger_kwargs
#     logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

#     td3(lambda : gym.make(args.env), actor_critic=core.MLPActorCritic,
#         ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
#         gamma=args.gamma, seed=args.seed, epochs=args.epochs,
#         logger_kwargs=logger_kwargs)