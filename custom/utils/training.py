import numpy as np


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