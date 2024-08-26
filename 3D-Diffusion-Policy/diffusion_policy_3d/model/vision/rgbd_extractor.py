import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import time

from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from .DFormer import DFormer_Base, DFormer_Large, DFormer_Small, DFormer_Tiny

def create_mlp(
        input_dim: int,
        output_dim: int,
        net_arch: List[int],
        activation_fn: Type[nn.Module] = nn.ReLU,
        squash_output: bool = False,
) -> List[nn.Module]:
    """
    Create a multi layer perceptron (MLP), which is
    a collection of fully-connected layers each followed by an activation function.

    :param input_dim: Dimension of the input vector
    :param output_dim:
    :param net_arch: Architecture of the neural net
        It represents the number of units per layer.
        The length of this list is the number of layers.
    :param activation_fn: The activation function
        to use after each layer.
    :param squash_output: Whether to squash the output using a Tanh
        activation function
    :return:
    """

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules



class RGBDEncoder(nn.Module):
    def __init__(self,
                 rgbd_network_backbone: str='DFormer_Base',
                 out_channels: int=1024,
                 use_layernorm: bool=False,
                 final_norm: str='none',
                 **kwargs
                 ):
        super().__init__()
        block_channel = [64, 128, 256]

        if rgbd_network_backbone == 'DFormer_Tiny':
            in_channels = 384
        elif rgbd_network_backbone == 'DFormer_Small':
            in_channels = 768
        elif rgbd_network_backbone == 'DFormer_Base':
            in_channels = 768
        elif rgbd_network_backbone == 'DFormer_Large':
            in_channels = 864
        else:
            raise NotImplementedError(f"RGBDEncoder backbone only supports 4 types, but got {rgbd_network_backbone}")

        cprint("[RGBDEncoder] use_layernorm: {}".format(use_layernorm), 'cyan')
        cprint("[RGBDEncoder] use_final_norm: {}".format(final_norm), 'cyan')
        cprint(f"[RGBDEncoder] input channals: {in_channels}", 'cyan')

        self.mlp = nn.Sequential(
            nn.Linear(in_channels, block_channel[0]),
            nn.LayerNorm(block_channel[0]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[0], block_channel[1]),
            nn.LayerNorm(block_channel[1]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
            nn.Linear(block_channel[1], block_channel[2]),
            nn.LayerNorm(block_channel[2]) if use_layernorm else nn.Identity(),
            nn.ReLU(),
        )
        
        
        if final_norm == 'layernorm':
            self.final_projection = nn.Sequential(
                nn.Linear(block_channel[-1], out_channels),
                nn.LayerNorm(out_channels)
            )
        elif final_norm == 'none':
            self.final_projection = nn.Linear(block_channel[-1], out_channels)
        else:
            raise NotImplementedError(f"final_norm: {final_norm}")
         
         
    def forward(self, x):
        x = self.mlp(x)
        # x = torch.max(x, 1)[0]
        x = self.final_projection(x)
        return x


class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 rgbd_network_backbone='DFormer_Base',
                 rgbd_encoder_cfg = None
                 ):
        super().__init__()
        self.depth_image_key = 'depth'
        self.state_key = 'agent_pos'
        self.rgb_image_key = 'img'
        self.n_output_channels = out_channel
        
        self.rgb_image_shape = observation_space[self.rgb_image_key]
        self.depth_image_shape = observation_space[self.depth_image_key]
        self.state_shape = observation_space[self.state_key]

            
        
        
        cprint(f"[DP3Encoder] rgb image shape: {self.rgb_image_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint(f"[DP3Encoder] depth image shape: {self.depth_image_shape}", "yellow")
        cprint("[DP3Encoder] rgbd_network_backbone: {}".format(rgbd_network_backbone), 'yellow')
        

        self.rgbd_network_backbone = rgbd_network_backbone
        if rgbd_network_backbone == 'DFormer_Tiny':
            self.extractor = DFormer_Tiny(pretrained = True)
        elif rgbd_network_backbone == 'DFormer_Small':
            self.extractor = DFormer_Small(pretrained = True)
        elif rgbd_network_backbone == 'DFormer_Base':
            self.extractor = DFormer_Base(pretrained = True)
        elif rgbd_network_backbone == 'DFormer_Large':
            self.extractor = DFormer_Large(pretrained = True)
        else:
            raise NotImplementedError(f"rgbd_network_backbone: {rgbd_network_backbone}")
        del self.extractor.pred
        self.extractor.eval()
        # self.extractor.requires_grad_(False)
        for p in self.extractor.parameters():
            p.requires_grad = False
        

        self.rgbd_encoder = RGBDEncoder(**rgbd_encoder_cfg)
        if len(state_mlp_size) == 0:
            raise RuntimeError(f"State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.n_output_channels  += output_dim
        self.state_mlp = nn.Sequential(*create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn))

        cprint(f"[DP3Encoder] output dim: {self.n_output_channels}", "red")


    def forward(self, observations: Dict) -> torch.Tensor:
        img = observations[self.rgb_image_key]
        depth = observations[self.depth_image_key]
        vision = torch.cat([img,depth],dim=1)
        assert len(vision.shape) == 4, cprint(f"vision shape: {vision.shape}, length should be 4", "red")
        with torch.no_grad():
            vision_feat = self.extractor(vision)    # B * out_channel
        rgbd_feat = self.rgbd_encoder(vision_feat)
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([rgbd_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels