from lib.datasets.dataset_skels_test import skels_dataset_test
from lib.arch.SegNet3D import UNet

import torch
from torch.utils.data import DataLoader

from empatches import EMPatches
import numpy as np
import tifffile as tif
import os

from ryu_pytools import tensor_to_ndarr, arr_info, check_dir

device = torch.device('cuda:0')
model = UNet(dim=3)
model.to(device)
ckpt_path = '/home/ryuuyou/E5/project/seg_unet_cldice/out/weights/SegNet3D_cldice/Epoch_090.pth'
# ckpt_path = '/home/ryuuyou/E5/project/seg_unet_cldice/out/weights/SegNet3D_V4/Epoch_2500.pth'
ckpt = torch.load(ckpt_path)
model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})

# data_path = '/home/ryuuyou/E5/project/data/skels_aug/img_deconv'
data_path = '/home/ryuuyou/E5/project/data/skels_test/img_ori'
ds = skels_dataset_test(data_path, patch_size=128, num=30)
dl = DataLoader(ds, batch_size=1, shuffle=False)

emp = EMPatches()
# save_path = '/home/ryuuyou/E5/project/data/skels_aug/result/unet/deconv/'
save_path = '/home/ryuuyou/E5/project/data/skels_test/result/unet/cldice/'
check_dir(save_path)
for index, img in enumerate(dl):
    print(index)
    img = img.to(device)
    pred_mask = model(img)
    indices = ds.patches_indices
    mask_mp = emp.merge_patches(list(tensor_to_ndarr(pred_mask.squeeze(1))), indices)
    # mask_mp[mask_mp>=0.5]=1
    # mask_mp[mask_mp<0.5]=0
    tif.imwrite(os.path.join(save_path, f'pred_{ds.img_name_list[index].replace("img", "mask")}'), mask_mp)

