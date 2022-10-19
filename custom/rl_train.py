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

if __name__ == "__main__":

    encoder = load_observer("/home/rvarga/implementation/robomimic/custom/ckpt/epoch49.pth")
    encoder.set_eval()

    expert_data_path='/home/rvarga/implementation/robomimic/custom/data/extended_low_dim_shaped.hdf5'

    print("Loading the expert demonstration data into the replay buffer...")
    assert os.path.exists(expert_data_path)

    render = False

    # Setup environment based on the dataset
    env = setup_environment(encoder=encoder, hdf5_path=expert_data_path, render=render)

    data_loader = get_data_loader(dataset_path=expert_data_path, seq_length=1, normalize_obs=True)

    obs_normalization_stats = data_loader.dataset.get_obs_normalization_stats()

    data_loader_iterator = iter(data_loader)

    demo_data = next(data_loader_iterator)

    print("Environment set up")


    td3(env, 
        max_ep_len=200, 
        epochs=100,
        start_steps=2000,
        steps_per_epoch=1000,
        # ac_kwargs=dict(hidden_sizes=[400, 400, 300]),
        ac_kwargs=dict(hidden_sizes=[400, 300]),
        update_after=2000,
        update_every=50,
        polyak=0.95,
        num_test_episodes=1, 
        replay_size=int(1e6), 
        pretrain_on_demonstration=False, 
        pretrain_steps=50,
        encoder=None,
        # batch_size=400,
        batch_size=1000,
        force_scale=None,
        expert_data=None,
        obs_normalization_stats=obs_normalization_stats,
        # pi_lr=1e-3,
        # q_lr=1e-5,
        # pi_lr=1e-4,
        # q_lr=1e-5,
        pi_lr=1e-4,
        q_lr=1e-5*10,
        noise_clip=0.1,
        act_noise=0.02,
        policy_delay=5,
        render=render,
        target_noise=0.1)

    
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
