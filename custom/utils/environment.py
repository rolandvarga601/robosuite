from tabnanny import check
import robosuite as suite
from robosuite.wrappers import GymWrapper
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
from gym import Env


def setup_environment(render_kwargs, encoder=None, hdf5_path=None, keys=None):

    # Create dict to hold options that will be passed to env creation call
    options = {}

    # print welcome info
    print("Welcome to robosuite v{}!".format(suite.__version__))
    print(suite.__logo__)

    # # Choose environment and add it to options
    # # options["env_name"] = choose_environment()
    # options["env_name"] = "PickPlaceCan"

    # # We simply choose a single (single-armed) robot to instantiate in the environment
    # # options["robots"] = choose_robots(exclude_bimanual=True)
    # options["robots"] = "Panda"

    # # Hacky way to grab joint dimension for now
    # joint_dim = 6 if options["robots"] == "UR5e" else 7

    # # Choose controller
    # # controller_name = choose_controller()
    # controller_name = "OSC_POSE"

    # # Load the desired controller
    # options["controller_configs"] = suite.load_controller_config(default_controller=controller_name)

    # # Define the pre-defined controller actions to use (action_dim, num_test_steps, test_value)
    # controller_settings = {
    #     "OSC_POSE":         [6, 6, 0.1],
    #     "OSC_POSITION":     [3, 3, 0.1],
    #     "IK_POSE":          [6, 6, 0.01],
    #     "JOINT_POSITION":   [joint_dim, joint_dim, 0.2],
    #     "JOINT_VELOCITY":   [joint_dim, joint_dim, -0.1],
    #     "JOINT_TORQUE":     [joint_dim, joint_dim, 0.25]
    # }

    # # Define variables for each controller test
    # action_dim = controller_settings[controller_name][0]
    # num_test_steps = controller_settings[controller_name][1]
    # test_value = controller_settings[controller_name][2]

    # # Define the number of timesteps to use per controller action as well as timesteps in between actions
    # steps_per_action = 75
    # steps_per_rest = 75

    # Help message to user
    print()
    print("Press \"H\" to show the viewer control panel.")

    # create environment to use for online data gathering
    # hdf5_path = '/home/rvarga/implementation/robomimic/custom/data/extended_low_dim_shaped.hdf5'
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path=hdf5_path)
    # env = GymWrapper(
    #     EnvUtils.create_env_from_metadata(
    #         env_meta=env_meta,
    #         render=True,
    #         render_offscreen=False,
    #     ).env
    # )


    if encoder is not None:
        env = GymWrapper(
            EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=render_kwargs["onscreen"],
                render_offscreen=render_kwargs["offscreen"],
            ).env, 
            keys=[key.replace('object', 'object-state') for key in list(encoder.obs_key_shapes.keys())]
        )
    else:
        env = GymWrapper(
            EnvUtils.create_env_from_metadata(
                env_meta=env_meta,
                render=render_kwargs["onscreen"],
                render_offscreen=render_kwargs["offscreen"],
            ).env, 
            keys=keys
        )

    assert isinstance(env, Env)

    env.reset()

    if render_kwargs["onscreen"]:
        env.render()
        env.viewer.set_camera(camera_id=0)

    # initialize the task
    # env = GymWrapper(
    #     suite.make(
    #         **options,
    #         has_renderer=True,
    #         has_offscreen_renderer=False,
    #         ignore_done=True,
    #         use_camera_obs=False,
    #         horizon=(steps_per_action + steps_per_rest) * num_test_steps,
    #         control_freq=20,
    #         reward_shaping=True,
    #     )
    # )
    # env.reset()

    return env

