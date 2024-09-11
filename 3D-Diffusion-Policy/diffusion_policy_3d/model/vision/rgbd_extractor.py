import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import copy
import time
import sys
sys.path.append("/home/zxr/Documents/Github/DP_ur5e_open_door/3D-Diffusion-Policy/diffusion_policy_3d/model/vision")
# sys.path.insert(0, '/home/zxr/Documents/Github/DP_ur5e_open_door/3D-Diffusion-Policy/diffusion_policy_3d/model/vision')
from typing import Optional, Dict, Tuple, Union, List, Type
from termcolor import cprint
from DFormer import DFormer_Base, DFormer_Large, DFormer_Small, DFormer_Tiny

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




class QuickGELU(torch.nn.Module):
   def forward(self, x: torch.Tensor):
       return x * torch.sigmoid(1.702 * x)
   
class CLVEAttentiveLayer(nn.Module):
    def __init__(self, n_head, d_embed):
        super().__init__()
        self.layernorm_1 = nn.LayerNorm(d_embed)
        # self.attention = SelfAttention(n_head, d_embed)
        self.attention = nn.MultiheadAttention(d_embed, n_head, batch_first=True)
        self.layernorm_2 = nn.LayerNorm(d_embed)
        self.linear_1 = nn.Linear(d_embed, 4 * d_embed)
        self.linear_2 = nn.Linear(4 * d_embed, d_embed)
        self.activation = QuickGELU()

    def forward(self, x, causal_mask):
        residue = x
        x = self.layernorm_1(x)
        # x = self.attention(x, causal_mask = True)
        x, _ = self.attention(x, x, x, is_causal = True, attn_mask = causal_mask) if causal_mask is not None else self.attention(x, x, x)
        x += residue
        residue = x
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = self.activation(x)
        x = self.linear_2(x)
        x += residue
        return x
   
class CLVEProjectionHead(nn.Module):
    def __init__(self, input_dim, output_dim, dropout):
        super().__init__()
        input_dim = input_dim
        self.projection = nn.Linear(input_dim, output_dim)
        self.gelu = QuickGELU()
        self.fc = nn.Linear(output_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)

    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x += projected
        x = self.layer_norm(x)
        return x 

class CLVE(nn.Module):
    def __init__(self, 
                 rgbd_encoder_cfg = None
                 ):
        super().__init__()
        self.rgbd_network_backbone = rgbd_encoder_cfg.rgbd_network_backbone
        if self.rgbd_network_backbone == 'DFormer_Tiny':
            self.extractor = DFormer_Tiny(pretrained = True)
            self.d_embed = 384
        elif self.rgbd_network_backbone == 'DFormer_Small':
            self.extractor = DFormer_Small(pretrained = True)
            self.d_embed = 768
        elif self.rgbd_network_backbone == 'DFormer_Base':
            self.extractor = DFormer_Base(pretrained = True)
            self.d_embed = 768
        elif self.rgbd_network_backbone == 'DFormer_Large':
            self.extractor = DFormer_Large(pretrained = True)
            self.d_embed = 864
        else:
            raise NotImplementedError(f"rgbd_network_backbone: {self.rgbd_network_backbone}")
        del self.extractor.pred
        self.extractor.eval()
        # self.extractor.requires_grad_(False)
        for p in self.extractor.parameters():
            p.requires_grad = False
        
        self.image_size = 300
        self.attention_layers = nn.ModuleList([
            CLVEAttentiveLayer(rgbd_encoder_cfg.num_heads, self.d_embed) 
            for i in range(rgbd_encoder_cfg.num_layers_clve_attentive)])
        self.linear = nn.Linear(self.image_size , 1)
        self.projection_head = CLVEProjectionHead(input_dim=self.d_embed, output_dim=rgbd_encoder_cfg.out_channels, dropout=rgbd_encoder_cfg.dropout)


    def forward(self, rgbd) -> torch.Tensor:
        with torch.no_grad():
            rgbd_feat = self.extractor(rgbd)
        batch, channels, h, w = rgbd_feat.shape[0], rgbd_feat.shape[1], rgbd_feat.shape[2], rgbd_feat.shape[3]
        rgbd_feat = rgbd_feat.view(batch, channels, h *w).swapaxes(1,2)
        for layer in self.attention_layers:
            state = layer(rgbd_feat, None)   # causal mask not needed for image features
        out = self.linear(state.swapaxes(2,1)).squeeze(-1)
        out = self.projection_head(out)
        # out = out.swapaxes(1,2).mean([-1])
        return out

class CLVEModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image_encoder = CLVE(cfg)
        self.epoch = 0
        self.step = 0

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(path))

    def forward(self, image_a, image_b):
        image_embed_a = self.image_encoder(image_a)
        image_embed_b = self.image_encoder(image_b)
        return image_embed_a, image_embed_b

class DP3Encoder(nn.Module):
    def __init__(self, 
                 observation_space: Dict, 
                 out_channel=256,
                 state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU,
                 rgbd_network_backbone='DFormer_Base',
                 rgbd_encoder_cfg = None
                 ):
        super().__init__()
        self.top_rgb_image_key = 'top_img'
        self.top_depth_image_key = 'top_depth'
        self.right_rgb_image_key = 'right_img'
        self.right_depth_image_key = 'right_depth'
        self.state_key = 'agent_pos'
        
        self.n_output_channels = out_channel
        self.camera_num = 2
        
        self.top_rgb_image_shape = observation_space[self.top_rgb_image_key]
        self.right_rgb_image_shape = observation_space[self.right_rgb_image_key]
        self.top_depth_image_shape = observation_space[self.top_depth_image_key]
        self.right_depth_image_shape = observation_space[self.right_depth_image_key]
        self.state_shape = observation_space[self.state_key]

            
        
        
        cprint(f"[DP3Encoder] top rgb image shape: {self.top_rgb_image_shape}", "yellow")
        cprint(f"[DP3Encoder] right rgb image shape: {self.right_rgb_image_shape}", "yellow")
        cprint(f"[DP3Encoder] top depth image shape: {self.top_depth_image_shape}", "yellow")
        cprint(f"[DP3Encoder] right depth image shape: {self.right_depth_image_shape}", "yellow")
        cprint(f"[DP3Encoder] state shape: {self.state_shape}", "yellow")
        cprint("[DP3Encoder] rgbd_network_backbone: {}".format(rgbd_network_backbone), 'yellow')
        

        # self.rgbd_network_backbone = rgbd_network_backbone
        # if rgbd_network_backbone == 'DFormer_Tiny':
        #     self.extractor = DFormer_Tiny(pretrained = True)
        # elif rgbd_network_backbone == 'DFormer_Small':
        #     self.extractor = DFormer_Small(pretrained = True)
        # elif rgbd_network_backbone == 'DFormer_Base':
        #     self.extractor = DFormer_Base(pretrained = True)
        # elif rgbd_network_backbone == 'DFormer_Large':
        #     self.extractor = DFormer_Large(pretrained = True)
        # else:
        #     raise NotImplementedError(f"rgbd_network_backbone: {rgbd_network_backbone}")
        # del self.extractor.pred
        # self.extractor.eval()
        # # self.extractor.requires_grad_(False)
        # for p in self.extractor.parameters():
        #     p.requires_grad = False
        

        self.rgbd_encoder = CLVE(**rgbd_encoder_cfg)
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
        top_img = observations[self.top_rgb_image_key]
        top_depth = observations[self.top_depth_image_key]
        right_img = observations[self.right_rgb_image_key]
        right_depth = observations[self.right_depth_image_key]
        top_vision = torch.cat([top_img,top_depth],dim=1)
        right_vision = torch.cat([right_img,right_depth], dim=1)
        assert len(top_vision.shape) == 4, cprint(f"top vision shape: {top_vision.shape}, length should be 4", "red")
        assert len(right_vision.shape) == 4, cprint(f"right vision shape: {right_vision.shape}, length should be 4", "red")
        with torch.no_grad():
            top_vision_feat = self.extractor(top_vision)    # B * Dformer_out_dim
            right_vision_feat = self.extractor(right_vision)    # B * Dformer_out_dim
        rgbd_feat = self.rgbd_encoder(torch.cat([top_vision_feat, right_vision_feat], dim=-1)) # B* encoder_out_dim
            
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * encoder_out_dim
        final_feat = torch.cat([rgbd_feat, state_feat], dim=-1)
        return final_feat


    def output_shape(self):
        return self.n_output_channels