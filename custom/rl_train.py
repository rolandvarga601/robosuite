"""
This is a custom demo code to train a reinforcement learning based controller.

Please reference the documentation of Controllers in the Modules section for an overview of each controller.
Controllers are expected to behave in a generally controlled manner, according to their control space. The expected
sequential qualitative behavior during the test is described below for each controller:

* OSC_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
            z-axis, relative to the global coordinate frame
* OSC_POSITION: Gripper moves sequentially and linearly in x, y, z direction, relative to the global coordinate frame
* IK_POSE: Gripper moves sequentially and linearly in x, y, z direction, then sequentially rotates in x-axis, y-axis,
            z-axis, relative to the local robot end effector frame
* JOINT_POSITION: Robot Joints move sequentially in a controlled fashion
* JOINT_VELOCITY: Robot Joints move sequentially in a controlled fashion
* JOINT_TORQUE: Unlike other controllers, joint torque controller is expected to act rather lethargic, as the
            "controller" is really just a wrapper for direct torque control of the mujoco actuators. Therefore, a
            "neutral" value of 0 torque will not guarantee a stable robot when it has non-zero velocity!

"""

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.utils.input_utils import *
from robosuite.robots import Bimanual

from utils.environment import setup_environment
from algos.agent import RandomAgent
from utils.training import rollout
from algos.td3 import td3
from utils.encoding import load_observer
import os
from utils.training import get_data_loader
from robomimic.config import config_factory
import robomimic.utils.obs_utils as ObsUtils

from utils.encoding import encode_obs

from collections import OrderedDict

import torch


if __name__ == "__main__":

    # expert_data_path='/home/rvarga/implementation/robomimic/custom/data/extended_low_dim_shaped.hdf5'
    # expert_data_path='/home/rvarga/implementation/robomimic/datasets/lift/mg/low_dim_shaped.hdf5'

    seed = 22
    
    expert_data_path = OrderedDict()
    # expert_data_path['success']='/home/rvarga/implementation/robomimic/datasets/lift/ph/low_dim_shaped_donemode0.hdf5'
    # expert_data_path['exp']='/home/rvarga/implementation/robomimic/datasets/lift/mg/low_dim_shaped_donemode0.hdf5'
    expert_data_path['success']='/home/rvarga/implementation/robomimic/datasets/lift/ph/low_dim_donemode0.hdf5'
    expert_data_path['exp']='/home/rvarga/implementation/robomimic/datasets/lift/mg/low_dim_donemode0.hdf5'

    render_kwargs = dict()
    render_kwargs["onscreen"] = False
    render_kwargs["offscreen"] = True
    fix_scenario = False
    use_encoder = False
    # keys = ['object-state', 'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos']
    # keys = ['object-state', 'robot0_gripper_qpos', 'robot0_gripper_qvel']
    keys = ['object-state', 'robot0_gripper_qpos', 'robot0_eef_quat']
    reward_correction = None
    success_boost = 5*0
    # target_bounds = {"lb" : 0, "ub" : 1}
    target_bounds = {"lb" : 0}
    do_underestimation_step = True


    if use_encoder:
        encoder = load_observer("/home/rvarga/implementation/robomimic/custom/ckpt/epoch99.pth", dataset_path=expert_data_path['exp'])
        encoder.set_eval()
    else:
        encoder = None
        config = config_factory(algo_name="vae_rep")

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        ObsUtils.initialize_obs_utils_with_config(config)

    assert os.path.exists(expert_data_path['exp'])

    print("Environment set up")
    # Setup environment based on the dataset
    env = setup_environment(encoder=encoder, hdf5_path=expert_data_path['exp'], render_kwargs=render_kwargs, keys=keys)

    print("Loading the expert demonstration data...")

    data_loader = dict()
    if use_encoder:
        # Load normalized data
        data_loader["exp"] = get_data_loader(dataset_path=expert_data_path['exp'], seq_length=1, normalize_obs=True)
        # data_loader["success"] = get_data_loader(dataset_path=expert_data_path['success'], seq_length=1, normalize_obs=False)
    else:
        # data_loader['exp'] = get_data_loader(dataset_path=expert_data_path['exp'], seq_length=1, normalize_obs=False)
        data_loader['exp'] = get_data_loader(dataset_path=expert_data_path['exp'], seq_length=1, normalize_obs=True)
        # data_loader['success'] = get_data_loader(dataset_path=expert_data_path['success'], seq_length=1, normalize_obs=False)

    print("Expert data has been loaded")

    if use_encoder:
        obs_normalization_stats = data_loader['exp'].dataset.get_obs_normalization_stats()
    else:
        # obs_normalization_stats = None
        obs_normalization_stats = data_loader['exp'].dataset.get_obs_normalization_stats()


    data_loader_iterator = dict()
    data_loader_iterator['exp'] = iter(data_loader['exp'])
    # data_loader_iterator['success'] = iter(data_loader['success'])

    demo_data = OrderedDict()
    # demo_data['success'] = next(data_loader_iterator['success'])
    demo_data['exp'] = next(data_loader_iterator['exp'])

    if success_boost is not None:
        demo_data['exp']['rewards'][demo_data['exp']['dones'] > 0] += success_boost

    # if use_encoder:
    #     obs_norms = []
    #     for k in range(data_loader['success'].batch_size):
    #         obs_dict = {key: torch.unsqueeze(demo_data['success']["obs"][key][k, 0, :], 0) for key in demo_data['success']["obs"]}
    #         obs_norms.append(ObsUtils.normalize_obs(obs_dict=obs_dict, obs_normalization_stats=obs_normalization_stats))
        
    #     demo_data['success']["obs"] = {key: torch.cat(tuple(torch.unsqueeze(o[key], 0) for o in obs_norms), 0) for key in obs_dict}

    #     obs_norms = []
    #     for k in range(data_loader['success'].batch_size):
    #         obs_dict = {key: torch.unsqueeze(demo_data['success']["next_obs"][key][k, 0, :], 0) for key in demo_data['success']["next_obs"]}
    #         obs_norms.append(ObsUtils.normalize_obs(obs_dict=obs_dict, obs_normalization_stats=obs_normalization_stats))
        
    #     demo_data['success']["next_obs"] = {key: torch.cat(tuple(torch.unsqueeze(o[key], 0) for o in obs_norms), 0) for key in obs_dict}

    td3(env, 
        max_ep_len=200, 
        epochs=1000,
        start_steps=0,
        steps_per_epoch=1000,
        # ac_kwargs=dict(hidden_sizes=[400, 400, 300]),
        # ac_kwargs=dict(hidden_sizes=[256, 256]),
        # ac_kwargs=dict(hidden_sizes=[128, 128]),
        # ac_kwargs=dict(hidden_sizes=[84, 84]),
        ac_kwargs=dict(hidden_sizes=[64, 64]),
        # ac_kwargs=dict(hidden_sizes=[22, 22, 22]),
        update_after=0,
        update_every=10,
        polyak=0.995*0+0.995,
        gamma=0.9*0+0.99,
        num_test_episodes=1, 
        replay_size=int(1e6)*0+int(250000), 
        pretrain_on_demonstration=True, 
        pretrain_steps=4000,
        encoder=encoder,
        # batch_size=400,
        batch_size=4000,
        force_scale=None,
        expert_data_dict=demo_data,
        expert_data_path=expert_data_path,
        obs_normalization_stats=obs_normalization_stats,
        # pi_lr=1e-3,
        # q_lr=1e-5,
        # pi_lr=1e-4,
        # q_lr=1e-5,
        pi_lr=1e-4*10*10/10/2/4,
        q_lr=1e-3*10/10/2/4,
        noise_clip=0.5,
        act_noise=0.3,
        policy_delay=2*0+10*0+2,
        target_noise=0.2,
        fix_scenario=fix_scenario,
        reward_correction=reward_correction,
        success_boost=success_boost,
        seed=seed,
        target_bounds=target_bounds,
        do_underestimation_step=do_underestimation_step)

    
    # Paramset that is almost working.... 
    #
    # td3(env, 
    #     max_ep_len=200, 
    #     epochs=100,
    #     start_steps=1000,
    #     steps_per_epoch=1000,
    #     ac_kwargs=dict(hidden_sizes=[400, 400, 300]),
    #     update_after=1000,
    #     update_every=50,
    #     num_test_episodes=1, 
    #     replay_size=int(1e6), 
    #     pretrain_on_demonstration=False, 
    #     pretrain_steps=50,
    #     encoder=encoder,
    #     batch_size=800,
    #     force_scale=None,
    #     expert_data=None,
    #     obs_normalization_stats=obs_normalization_stats,
    #     pi_lr=1e-4,
    #     q_lr=1e-4,
    #     noise_clip=0.3,
    #     act_noise=0.1,
    #     policy_delay=2,
    #     render=render)

    print("Starting rollout")

    # rollout_data = rollout(env=env, agent=agent, max_path_length=300, render=True, render_kwargs=None)


    # Shut down this env before starting the next test
    # env.close()
