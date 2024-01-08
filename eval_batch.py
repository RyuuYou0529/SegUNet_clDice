from lib.datasets.dataset_skels_test import skels_dataset_test
from lib.arch.SegNet3D import UNet
from lib.utils.utils import tensor_to_ndarr, check_dir

import torch
from torch.utils.data import DataLoader

from empatches import EMPatches
import tifffile as tif
from tqdm import tqdm
import os

if __name__ == '__main__':
    device = torch.device('cuda:0')
    model = UNet(features=[32,64,128], dim=3)
    model.to(device)
    ckpt_path = 'out/weights/SegNet3D/Epoch_060.pth'
    ckpt = torch.load(ckpt_path)
    model.load_state_dict({k.replace('module.',''):v for k,v in ckpt['model'].items()})
    model.eval()

    data_path = 'data/mouse_test/img'
    ds = skels_dataset_test(data_path, patch_size=128, use_preprocess=True)
    dl = DataLoader(ds, batch_size=1, shuffle=False)

    emp = EMPatches()
    save_path = 'data/mouse_test/result/segnet3d'
    check_dir(save_path)
    for index, img in tqdm(enumerate(dl)):
        img = img.to(device)
        pred_mask = model(img)
        indices = ds.patches_indices
        mask_mp = emp.merge_patches(list(tensor_to_ndarr(pred_mask.squeeze(1))), indices)
        # mask_mp[mask_mp>=0.5]=1
        # mask_mp[mask_mp<0.5]=0
        tif.imwrite(os.path.join(save_path, f'pred_{ds.img_name_list[index].replace("img", "mask")}'), mask_mp)

