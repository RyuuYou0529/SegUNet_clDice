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
model = UNet(features=[64,128,256], dim=3)
model.to(device)
ckpt_path = 'out/weights/SegNet3D_lite_mouse/Epoch_120.pth'
ckpt = torch.load(ckpt_path)
model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})
model.eval()

data_path = 'data/mouse_test/img'
ds = skels_dataset_test(data_path, patch_size=128, use_preprocess=True)
dl = DataLoader(ds, batch_size=1, shuffle=False)

emp = EMPatches()
save_path = 'data/mouse_test/result/lite'
check_dir(save_path)
for index, img in tqdm(enumerate(dl)):
    img = img.to(device)
    pred_mask = model(img)
    indices = ds.patches_indices
    mask_mp = emp.merge_patches(list(tensor_to_ndarr(pred_mask.squeeze(1))), indices)
    # mask_mp[mask_mp>=0.5]=1
    # mask_mp[mask_mp<0.5]=0
    tif.imwrite(os.path.join(save_path, f'pred_{ds.img_name_list[index].replace("img", "mask")}'), mask_mp)

