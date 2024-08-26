import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


_transforms = transforms.Compose([
        transforms.Resize([224,224]),
        # transforms.CenterCrop(224)
        ])


class ReplayBuffer:
    """
    Zarr-based temporal datastructure.
    Assumes first dimension to be time. Only chunk in time dimension.
    """
    def __init__(self,
                 data_path = '/home/zxr/Documents/Github/DP_ur5e_open_door/3D-Diffusion-Policy/data/real_door'):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        # assert('data' in root)
        # assert('meta' in root)
        # assert('episode_ends' in root['meta'])
        # for key, value in root['data'].items():
        #     assert(value.shape[0] == root['meta']['episode_ends'][-1])
        # self.root = root
        self.data, self.episode_ends = self.get_data(data_path)


    def get_data(self,data_dir_path_abs):
        data = dict()
        img_ls = []
        state_ls = []
        depth_ls = []
        action_ls = []
        episode_end_ls = [0]
        for dir in os.listdir(data_dir_path_abs):
            obs = np.load(os.path.join(data_dir_path_abs, dir, 'obs.npy'),allow_pickle=True)[:-1]
            action_tmp = np.load(os.path.join(data_dir_path_abs, dir, 'action_list.npy'),allow_pickle=True)
            episode_end_ls.append(episode_end_ls[-1]+len(obs))
            for i in range(len(obs)):
                img_ls.append(np.array(_transforms(Image.fromarray(cv2.cvtColor(obs[i]['img'],cv2.COLOR_BGR2RGB)))).transpose(2,0,1))
                depth_ls.append(np.array(_transforms(Image.fromarray(obs[i]['depth'],'L'))).reshape(-1,224,224))
                state_ls.append(obs[i]['agent_pos'])
                action_ls.append(action_tmp[i])
        img = np.array(img_ls)
        state = np.array(state_ls)
        depth = np.array(depth_ls)
        action = np.array(action_ls)
        data['state'] = state
        data['img'] = img
        data['depth'] = depth
        data['action'] = action
        return data, np.array(episode_end_ls,dtype='int64')[1:]
    
    @property
    def episode_lengths(self):
        ends = self.episode_ends[:]
        ends = np.insert(ends, 0, 0)
        lengths = np.diff(ends)
        return lengths
    
    @property
    def n_episodes(self):
        return len(self.episode_ends)
    
    @property
    def n_steps(self):
        if len(self.episode_ends) == 0:
            return 0
        return self.episode_ends[-1]
    
    def keys(self):
        return self.data.keys()
    
    def values(self):
        return self.data.values()
    
    def items(self):
        return self.data.items()
    
    def __getitem__(self, key):
        return self.data[key]

    def __contains__(self, key):
        return key in self.data


if __name__ == "__main__":
    replaybuffer = ReplayBuffer()

    print('a')
