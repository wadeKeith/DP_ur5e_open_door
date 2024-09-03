import numpy as np
import os
import cv2
from PIL import Image



global_address = '/home/zxr/Documents/Github/DP_ur5e_open_door/3D-Diffusion-Policy/data/vision_data'
class0_img_ls = []
class0_depth_ls = []
class1_img_ls = []
class1_depth_ls = []
class2_img_ls = []
class2_depth_ls = []
class3_img_ls = []
class3_depth_ls = []

for file in os.listdir(global_address):
    if 'npy' in file:
        rgbd3 = np.load(os.path.join(global_address, file), allow_pickle=True).item()
        class3_img_ls.append(rgbd3['img_ls'])
        class3_depth_ls.append(rgbd3['depth_ls'])
    else:
        rgbd0 = np.load(os.path.join(global_address, file, 'class0.npy'), allow_pickle=True).item()
        rgbd1 = np.load(os.path.join(global_address, file, 'class1.npy'), allow_pickle=True).item()
        rgbd2 = np.load(os.path.join(global_address, file, 'class2.npy'), allow_pickle=True).item()
        class0_img_ls.append(rgbd0['img_ls'])
        class0_depth_ls.append(rgbd0['depth_ls'])
        class1_img_ls.append(rgbd1['img_ls'])
        class1_depth_ls.append(rgbd1['depth_ls'])
        class2_img_ls.append(rgbd2['img_ls'])
        class2_depth_ls.append(rgbd2['depth_ls'])

class0_img_ls = np.concatenate(class0_img_ls,axis=0)
class0_depth_ls = np.concatenate(class0_depth_ls, axis=0)
class1_img_ls = np.concatenate(class1_img_ls,axis=0)
class1_depth_ls = np.concatenate(class1_depth_ls, axis=0)
class2_img_ls = np.concatenate(class2_img_ls,axis=0)
class2_depth_ls = np.concatenate(class2_depth_ls, axis=0)
class3_img_ls = np.concatenate(class3_img_ls,axis=0)
class3_depth_ls = np.concatenate(class3_depth_ls, axis=0)


print('a')






