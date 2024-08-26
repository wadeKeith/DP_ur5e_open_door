from diffusion_policy_3d.dataset.base_dataset import BaseDataset
import hydra
from diffusion_policy_3d.dataset.adroit_dataset import AdroitDataset

dataset = AdroitDataset('data/adroit_door_expert.zarr',4,1,3,42,0.02,90)

test = dataset[0]


print('a')









