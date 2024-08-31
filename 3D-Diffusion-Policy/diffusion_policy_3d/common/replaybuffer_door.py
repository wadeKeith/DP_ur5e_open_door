import os
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


def find_yellow_region(image, min_shape = [224,224]):
    
    # 转换为HSV色彩空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 定义黄色的HSV范围
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # 根据HSV范围构建掩膜
    mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # 找到所有的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 如果找到了轮廓
    if contours:
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 得到边界框
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # 裁剪出图像中的黄色区域
        # cropped_image = image[y:y+h, x:x+w]
        
        if w<min_shape[0] or h<min_shape[1]:
            return [0,0,image.shape[1], image.shape[0]]
        else:
            return [x, y, w, h]
    else:
        return [0,0,image.shape[1], image.shape[0]]
    


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
                 data_path = '/home/zxr/Documents/Github/DP_ur5e_open_door/3D-Diffusion-Policy/data/real_door',bbox= [87,80,500,400] ):
        """
        Dummy constructor. Use copy_from* and create_from* class methods instead.
        """
        # assert('data' in root)
        # assert('meta' in root)
        # assert('episode_ends' in root['meta'])
        # for key, value in root['data'].items():
        #     assert(value.shape[0] == root['meta']['episode_ends'][-1])
        # self.root = root
        self.top_bbox = bbox # [x1, y1, w, h]
        self.data, self.episode_ends = self.get_data(data_path)


    def get_data(self,data_dir_path_abs):
        data = dict()
        top_img_ls = []
        right_img_ls = []
        state_ls = []
        top_depth_ls = []
        right_depth_ls = []
        action_ls = []
        episode_end_ls = [0]
        for dir in os.listdir(data_dir_path_abs):
            obs = np.load(os.path.join(data_dir_path_abs, dir, 'obs.npy'),allow_pickle=True)[:-1]
            action_tmp = np.load(os.path.join(data_dir_path_abs, dir, 'action_list.npy'),allow_pickle=True)
            episode_end_ls.append(episode_end_ls[-1]+len(obs))
            for i in range(len(obs)):
                top_img_ls.append(np.array(_transforms(Image.fromarray(cv2.cvtColor(obs[i]['top_img'],cv2.COLOR_BGR2RGB)[self.top_bbox[1]:self.top_bbox[1]+self.top_bbox[3], self.top_bbox[0]:self.top_bbox[0]+self.top_bbox[2],:]))).transpose(2,0,1))
                top_depth_ls.append(np.array(_transforms(Image.fromarray(obs[i]['top_depth'][self.top_bbox[1]:self.top_bbox[1]+self.top_bbox[3], self.top_bbox[0]:self.top_bbox[0]+self.top_bbox[2]],'L'))).reshape(-1,224,224))
                right_bbox = find_yellow_region(obs[i]['right_img'])
                right_img_ls.append(np.array(_transforms(Image.fromarray(cv2.cvtColor(obs[i]['right_img'],cv2.COLOR_BGR2RGB)[right_bbox[1]:right_bbox[1]+right_bbox[3], right_bbox[0]:right_bbox[0]+right_bbox[2],:]))).transpose(2,0,1))
                right_depth_ls.append(np.array(_transforms(Image.fromarray(obs[i]['right_depth'][right_bbox[1]:right_bbox[1]+right_bbox[3], right_bbox[0]:right_bbox[0]+right_bbox[2]],'L'))).reshape(-1,224,224))
                state_ls.append(obs[i]['agent_pos'])
                action_ls.append(action_tmp[i])
        top_img = np.array(top_img_ls)
        right_img = np.array(right_img_ls)
        state = np.array(state_ls)
        top_depth = np.array(top_depth_ls)
        right_depth = np.array(right_depth_ls)
        action = np.array(action_ls)
        data['state'] = state
        data['top_img'] = top_img
        data['top_depth'] = top_depth
        data['right_img'] = right_img
        data['right_depth'] = right_depth
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
