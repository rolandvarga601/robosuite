import numpy as np
from robomimic.utils.dataset import SequenceDataset
from torch.utils.data import DataLoader


def get_data_loader(dataset_path, seq_length=1):
    """
    Get the dataset from hdf5 file
    Args:
        dataset_path (str): path to the dataset hdf5
    """
    dataset = SequenceDataset(
        hdf5_path=dataset_path,
        obs_keys=(                      # observations we want to appear in batches
            "robot0_eef_force",
            "robot0_eef_pos", 
            "robot0_eef_quat",
            "robot0_eef_vel_ang",
            "robot0_eef_vel_lin",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel", 
            "object",
        ),
        dataset_keys=(                  # can optionally specify more keys here if they should appear in batches
            "actions", 
            "rewards", 
            "dones",
        ),
        load_next_obs=True,
        frame_stack=1,
        seq_length=seq_length,                  # length-seq_length temporal sequences
        pad_frame_stack=True,
        pad_seq_length=True,            # pad last obs per trajectory to ensure all sequences are sampled
        get_pad_mask=False,
        goal_mode=None,
        hdf5_cache_mode="low_dim",          # cache everything from dataset except images in memory to avoid repeated file i/o
        hdf5_use_swmr=True,
        hdf5_normalize_obs=False,
        filter_by_attribute=None,       # can optionally provide a filter key here
    )
    print("\n============= Created Dataset =============")
    print(dataset)
    print("")

    data_loader = DataLoader(
        dataset=dataset,
        sampler=None,       # no custom sampling logic (uniform sampling)
        # sampler=SequentialSampler(dataset),       # no custom sampling logic (uniform sampling)
        batch_size=dataset.total_num_sequences,     # batches of size 100
        shuffle=False,
        num_workers=0,
        drop_last=True      # don't provide last batch in dataset pass if it's less than 100 in size
    )

    return data_loader

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def rollout(
        env,
        agent,
        max_path_length=np.inf,
        render=False,
        render_kwargs=None,
        video_writer=None,
):
    """
    Custom rollout function implemented in the robosuite-benchmark project that extends the basic rlkit functionality in the following ways:
    - Allows for automatic video writing if @video_writer is specified
    Added args:
        video_writer (imageio.get_writer): If specified, will write image frames to this writer
    The following is pulled directly from the rlkit rollout(...) function docstring:
    The following value for the following keys will be a 2D array, with the
    first dimension corresponding to the time dimension.
     - observations
     - actions
     - rewards
     - next_observations
     - terminals
    The next two elements will be lists of dictionaries, with the index into
    the list being the index into the time
     - agent_infos
     - env_infos
    """
    if render_kwargs is None:
        render_kwargs = {}
    observations = []
    actions = []
    rewards = []
    terminals = []
    agent_infos = []
    env_infos = []
    o = env.reset()
    print("Environment reset")
    agent.reset()
    print("Agent reset")
    next_o = None
    path_length = 0

    # Only render if specified AND there's no video writer
    if render and video_writer is None:
        env.render(**render_kwargs)

    while path_length < max_path_length:
        a, agent_info = agent.get_action(o)
        next_o, r, d, env_info = env.step(a)
        observations.append(o)
        rewards.append(r)
        terminals.append(d)
        actions.append(a)

        # Grab image data to write to video writer if specified
        if video_writer is not None:
            # We need to directly grab full observations so we can get image data
            full_obs = env._get_observation()

            # Grab image data (assume relevant camera name is the first in the env camera array)
            img = full_obs[env.camera_names[0] + "_image"]

            # Write to video writer
            video_writer.append_data(img[::-1])

        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1
        if d:
            break
        o = next_o
        if render:
            env.render(**render_kwargs)

    actions = np.array(actions)
    if len(actions.shape) == 1:
        actions = np.expand_dims(actions, 1)
    observations = np.array(observations)
    if len(observations.shape) == 1:
        observations = np.expand_dims(observations, 1)
        next_o = np.array([next_o])
    next_observations = np.vstack(
        (
            observations[1:, :],
            np.expand_dims(next_o, 0)
        )
    )
    return dict(
        observations=observations,
        actions=actions,
        rewards=np.array(rewards).reshape(-1, 1),
        next_observations=next_observations,
        terminals=np.array(terminals).reshape(-1, 1),
        agent_infos=agent_infos,
        env_infos=env_infos,
    )