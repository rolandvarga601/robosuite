import os
import robomimic.utils.obs_utils as ObsUtils
import robomimic.utils.file_utils as FileUtils
from robomimic.config import config_factory
from robomimic.algo import algo_factory
from robomimic.utils.file_utils import load_dict_from_checkpoint
import robomimic.utils.tensor_utils as TensorUtils
from robosuite.wrappers import GymWrapper
from collections import OrderedDict
import torch
import numpy as np


def encode_obs(encoder, obs):
    obs_list = OrderedDict()
    ptr = 0
    for key, size in encoder.obs_key_shapes.items():
        obs_list[key] = obs[ptr:ptr+size[0]]
        ptr = ptr + size[0]

    input_batch = dict()
    input_batch["obs"] = {k: torch.from_numpy(obs_list[k].reshape((1,obs_list[k].size))) for k in obs_list}
    batch_prep = TensorUtils.to_device(TensorUtils.to_float(input_batch), device='cuda')
    latent_obs = encoder.nets['policy'].nets['encoder'].forward(input=batch_prep['obs'])['mean']
    return np.squeeze(latent_obs.cpu().detach().numpy())


def get_model(dataset_path, device):
    """
    Use a default config to construct a VAE representation model.
    """

    # default BC config
    config = config_factory(algo_name="vae_rep")

    # read config to set up metadata for observation modalities (e.g. detecting rgb observations)
    ObsUtils.initialize_obs_utils_with_config(config)

    # read dataset to get some metadata for constructing model
    shape_meta = FileUtils.get_shape_metadata_from_dataset(
        dataset_path=dataset_path, 
        all_obs_keys=sorted((
            "robot0_eef_force",
            "robot0_eef_pos", 
            "robot0_eef_quat",
            "robot0_eef_vel_ang",
            "robot0_eef_vel_lin",
            "robot0_gripper_qpos",
            "robot0_gripper_qvel", 
            "object",
        )),
    )

    # make VAE model
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device,
    )
    return model


def load_observer(checkpoint_path):
    assert os.path.exists(checkpoint_path)

    model = get_model(dataset_path='/home/rvarga/implementation/robomimic/custom/data/extended_low_dim_shaped.hdf5',
        device='cuda')

    ckpt_dict = load_dict_from_checkpoint(checkpoint_path)

    model.deserialize(ckpt_dict["model"])
    
    return model