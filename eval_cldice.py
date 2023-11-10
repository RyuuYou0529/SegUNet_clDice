from lib.datasets.dataset_skels import skels_dataset
from lib.arch.SegNet3D import UNet

import torch
from torch.utils.data import DataLoader

from empatches import EMPatches
import numpy as np
import tifffile as tif
import os

from ryu_pytools import tensor_to_ndarr, arr_info

device = torch.device('cuda:0')
model = UNet(dim=3)
model.to(device)
ckpt_path = '/home/ryuuyou/E5/project/seg_unet_cldice/out/weights/SegNet3D_cldice/Epoch_2160.pth'
ckpt = torch.load(ckpt_path)
model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})

data_path = '/home/ryuuyou/E5/project/seg_unet_cldice/data/skels_test'
ds = skels_dataset(data_path, patch_size=128, use_preprocess=True)
dl = DataLoader(ds, batch_size=1, shuffle=False)

emp = EMPatches()
save_path = 'data/skels_test/res_v4/'
for index, (img, _) in enumerate(dl):
    print(index)
    img = img.to(device)
    pred_mask = model(img)
    indices = ds.patches_indices
    mask_mp = emp.merge_patches(list(tensor_to_ndarr(pred_mask.squeeze(1))), indices)
    # mask_mp[mask_mp>=0.5]=1
    # mask_mp[mask_mp<0.5]=0
    tif.imwrite(os.path.join(save_path, f'pred_{ds.mask_name_list[index]}'), mask_mp)

