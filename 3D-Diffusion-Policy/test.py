import torch
import os
from diffusion_policy_3d.model.vision.rgbd_extractor import CLVEModel
import os
import hydra
import torch
import dill
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
from termcolor import cprint
import shutil
import time
import threading
from hydra.core.hydra_config import HydraConfig
from diffusion_policy_3d.policy.dp3 import DP3
from diffusion_policy_3d.dataset.base_dataset import BaseDataset
from diffusion_policy_3d.env_runner.base_runner import BaseRunner
from diffusion_policy_3d.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy_3d.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy_3d.model.diffusion.ema_model import EMAModel
from diffusion_policy_3d.model.common.lr_scheduler import get_scheduler

import sys


OmegaConf.register_new_resolver("eval", eval, replace=True)


global_address = '/home/zxr/Documents/Github/DP_ur5e_open_door/3D-Diffusion-Policy/checkpoints'

sys.path.insert(0, '/home/zxr/Documents/Github/DP_ur5e_open_door/3D-Diffusion-Policy/diffusion_policy_3d/model/vision')


@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.joinpath(
        'diffusion_policy_3d', 'config'))
)
def main(cfg):
    rgbd_extractor = CLVEModel(cfg.policy.rgbd_encoder_cfg)
    checkpoints = torch.load(os.path.join(global_address, 'clve_last_step_ckpt.pth'))
    print('a')

if __name__ == "__main__":
    main()













