from lib.datasets.dataset_skels_test import skels_dataset_test
from lib.arch.SegNet3D import UNet

import torch
from torch.utils.data import DataLoader

from empatches import EMPatches
import numpy as np
import tifffile as tif
import os
from tqdm import tqdm

from ryu_pytools import tensor_to_ndarr, arr_info, check_dir

device = torch.device('cuda:0')
# model = UNet(features=[32,64,128], dim=3)
model = UNet(features=[64,128,256], dim=3)
model.to(device)
# ckpt_path = 'out/weights/SegNet3D_mini_mouse/Epoch_060.pth'
ckpt_path = 'out/weights/SegNet3D_lite_mouse/Epoch_160.pth'
ckpt = torch.load(ckpt_path)
model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})
model.eval()

img = tif.imread('data/mouse_aug/img_bg/img_35.tif').astype(np.float32)
img = torch.from_numpy(img)[None,None].to(device)

res = model(img)

tif.imwrite('res.tif', tensor_to_ndarr(res[0,0]))