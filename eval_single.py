from lib.arch.SegNet3D import UNet
from lib.datasets.dataset_skels_test import preprocess
from lib.utils.utils import check_dir, tensor_to_ndarr

import torch

import numpy as np
import tifffile as tif
import os

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = UNet(features=[32,64,128], dim=3)
    model.to(device)

    ckpt_path = 'out/weights/SegNet3D_mini_mouse/Epoch_200.pth'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})
    model.eval()

    data_path = 'data/mouse_test/img/img_21.tif'
    img = tif.imread(data_path).astype(np.float32)
    img = preprocess(img)
    img = torch.from_numpy(img)[None,None].to(device)

    res = model(img)

    save_path = './'
    check_dir(save_path)
    tif.imwrite(os.path.join(save_path, 'res.tif'), tensor_to_ndarr(res[0,0]))