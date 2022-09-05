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

if __name__ == "__main__":

    # Setup environment with hardcoded parameters
    env = setup_environment()

    print("Environment set up")

    # agent = RandomAgent(env.action_spec)
    # td3(setup_environment, max_ep_len=100)
    td3(env, max_ep_len=300, epochs=2, steps_per_epoch=3000, num_test_episodes=1, replay_size=int(1e6))

    print("Starting rollout")

    # rollout_data = rollout(env=env, agent=agent, max_path_length=300, render=True, render_kwargs=None)


    # Shut down this env before starting the next test
    # env.close()
