import torch
import algos.td3_core as core
from robomimic.config import config_factory
from utils.encoding import load_observer
import robomimic.utils.obs_utils as ObsUtils
from utils.environment import setup_environment
import os
import numpy as np
from utils.logger import MyLogger
from utils.training import get_data_loader
import json
from PIL import Image, ImageDraw, ImageFont

import imageio

if __name__ == "__main__":
    save_dir = '/home/rvarga/Data/Delft/thesis/implementation/robosuite/runs/lift-seed22-lb-underest-commonloss-200epochs'
    with open(os.path.join(save_dir, 'config.json'), 'r') as config_file:
        conf = json.load(config_file)

    expert_data_path = conf['expert_data_path']['exp']

    assert os.path.exists(expert_data_path)


    # ac_kwargs=dict(hidden_sizes=[64, 64])
    # render_kwargs = dict()
    # render_kwargs["onscreen"] = False
    # render_kwargs["offscreen"] = True
    # fix_scenario = False
    # use_encoder = False
    # keys = ['object-state', 'robot0_gripper_qpos', 'robot0_eef_quat']
    # reward_correction = None
    # success_boost = 0
    # seed = 42

    ac_kwargs=conf['ac_kwargs']
    render_kwargs = dict()
    render_kwargs["onscreen"] = False
    render_kwargs["offscreen"] = True
    fix_scenario = False
    use_encoder = conf['uses_encoder']
    keys = ['object-state', 'robot0_gripper_qpos', 'robot0_eef_quat']
    reward_correction = None
    success_boost = 0
    seed = conf['seed']*0+42

    # mylogger = MyLogger()

    if use_encoder:
        encoder = load_observer("/home/rvarga/implementation/robomimic/custom/ckpt/epoch99.pth", dataset_path=expert_data_path)
        encoder.set_eval()
    else:
        encoder = None
        config = config_factory(algo_name="vae_rep")

        # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
        ObsUtils.initialize_obs_utils_with_config(config)

        data_loader = get_data_loader(dataset_path=expert_data_path, seq_length=1, normalize_obs=True)

        obs_normalization_stats = data_loader.dataset.get_obs_normalization_stats()

    print("Environment set up")
    # Setup environment based on the dataset
    env = setup_environment(encoder=encoder, hdf5_path=expert_data_path, render_kwargs=render_kwargs, keys=keys)

    torch.manual_seed(seed)
    np.random.seed(seed)

    print("Creating experience collector environment")

    if encoder is not None:
        obs_dim = encoder.algo_config.vae["latent_dim"]
    else:
        obs_dim = env.observation_space.shape[0]
    
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # ckpt_folder = '/home/rvarga/Data/Delft/thesis/implementation/robosuite/custom/ckpt'
    ckpt_folder = os.path.join(save_dir, 'ckpt')
    epoch = 200

    # Create actor-critic module and target networks
    ac = core.MLPActorCritic(obs_dim, env.action_space, **ac_kwargs)
    ac.load_state_dict(torch.load(os.path.join(ckpt_folder, f"epoch{epoch}" + ".pth")))
    ac.eval()

    # set download folder and make it
    # download_folder = "/tmp/robomimic_ds_example"
    download_folder = save_dir
    os.makedirs(download_folder, exist_ok=True)

    # prepare to write playback trajectories to video
    video_path = os.path.join(download_folder, f"playback_epoch{epoch}.mp4")
    video_writer = imageio.get_writer(video_path, fps=20)

    max_ep_len = 150
    num_test_episodes = 100

    num_of_success = 0

    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", 24, encoding="unic")

    if obs_normalization_stats is not None:
        obs_stats = dict()
        obs_stats['mean'] = np.concatenate([obs_normalization_stats[key.replace('object-state','object')]['mean'][0] for key in env.keys])
        obs_stats['std'] = np.concatenate([obs_normalization_stats[key.replace('object-state','object')]['std'][0] for key in env.keys])

    for j in range(num_test_episodes):

        o, d, ep_ret, ep_len = env.reset(), False, 0, 0

        if obs_normalization_stats is not None:
            o = (o-obs_stats['mean']) / obs_stats['std']

        if env.unwrapped.has_renderer:
            env.render()
        elif env.unwrapped.has_offscreen_renderer:
            # video_img = env.render(mode="rgb_array", height=512, width=512, camera_name="agentview")
            video_img = env.sim.render(height=512, width=512, camera_name='frontview')[::-1]

            img = Image.fromarray(video_img)
            video_img_editable = ImageDraw.Draw(img)
            video_img_editable.text((15,15), f"Test {j+1}", font=font)
            video_img = np.array(img)

            video_writer.append_data(video_img)

        finish_delay = 20

        while not(finish_delay==0 or (ep_len == max_ep_len)):
            # Take deterministic actions at test time (noise_scale=0)
            a = ac.act(torch.as_tensor(o, dtype=torch.float32))
            # if d:
            #     a *= 0
            o, r, d, _ = env.step(a)

            if obs_normalization_stats is not None:
                o = (o-obs_stats['mean']) / obs_stats['std']

            d = int(env._check_success())

            ep_ret += r
            ep_len += 1

            if d:
                finish_delay -= 1

            if finish_delay == 0:
                num_of_success += 1

            if env.unwrapped.has_renderer:
                env.render()
            elif env.unwrapped.has_offscreen_renderer:
                # video_img = env.render(mode="rgb_array", height=512, width=512, camera_name="agentview")
                video_img = env.sim.render(height=512, width=512, camera_name='frontview')[::-1]

                img = Image.fromarray(video_img)
                video_img_editable = ImageDraw.Draw(img)
                video_img_editable.text((15,15), f"Test {j+1}", font=font)
                video_img = np.array(img)

                video_writer.append_data(video_img)

        # mylogger.store(TestEpRet=ep_ret, TestEpLen=ep_len)

        print(f"Episode return after test: {ep_ret}         Episode length: {ep_len}")

    print(f"Number of successful episodes: {num_of_success}/{num_test_episodes}")
    video_writer.close()



