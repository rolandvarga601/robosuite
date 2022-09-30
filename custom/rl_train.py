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

if __name__ == "__main__":

    encoder = load_observer("/home/rvarga/implementation/robomimic/custom/ckpt/epoch199.pth")
    encoder.set_eval()

    # Setup environment with hardcoded parameters
    env = setup_environment(encoder=encoder)

    print("Environment set up")

    # agent = RandomAgent(env.action_spec)
    # td3(setup_environment, max_ep_len=100)
    td3(env, 
        max_ep_len=200, 
        epochs=100,
        start_steps=0,
        steps_per_epoch=1000,
        update_after=1000, 
        num_test_episodes=1, 
        replay_size=int(1e6), 
        pretrain_on_demonstration=True, 
        pretrain_steps=40000,
        encoder=encoder,
        batch_size=120)

    print("Starting rollout")

    # rollout_data = rollout(env=env, agent=agent, max_path_length=300, render=True, render_kwargs=None)


    # Shut down this env before starting the next test
    # env.close()
