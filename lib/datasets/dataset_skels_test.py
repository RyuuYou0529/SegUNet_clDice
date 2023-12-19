from torch.utils.data import Dataset
import torch
import os

import tifffile as tiff
import numpy as np
from empatches import EMPatches

from tqdm import tqdm

def preprocess(img,percentiles=[0.01,0.9999]):
    # input img [0,65535]
    # output img [0,1]
    flattened_arr = np.sort(img.flatten())
    clip_low = int(percentiles[0] * len(flattened_arr))
    clip_high = int(percentiles[1] * len(flattened_arr))
    clipped_arr = np.clip(img, flattened_arr[clip_low], flattened_arr[clip_high])
    min_value = np.min(clipped_arr)
    max_value = np.max(clipped_arr) 
    img = (clipped_arr-min_value)/(max_value-min_value)
    return img

class skels_dataset_test(Dataset):
    def __init__(self, path, patch_size=64, overlap=0.0, use_preprocess=False, num=-1) -> None:
        super().__init__()
        print(f'patch_size: {patch_size}')
        print(f'overlap: {overlap}')

        img_path = path
        print(f'image path: {img_path}')

        emp = EMPatches()
        # patches
        self.img_patches_list = []
        # indices
        self.patches_indices = None
        # number of foreground points in each patches
        self.fgp_in_patches = []

        img_name_list = sorted(os.listdir(img_path))
        self.img_name_list = img_name_list

        for index, img_name in tqdm(enumerate(img_name_list[:num])):
            img = tiff.imread(os.path.join(img_path, img_name))
            img = img.astype(np.float32)

            # preprocess image
            if use_preprocess:
                img = preprocess(img)
            # extract pathces
            img_patches, img_indices = emp.extract_patches(img, patchsize=patch_size, overlap=overlap, stride=None, vox=True)
            # add patches
            self.img_patches_list = self.img_patches_list + img_patches

        # indices of pathces
        self.patches_indices = img_indices

        self.patches_per_volume = len(self.patches_indices)
        self.patches_num = len(self.img_patches_list)
        
        self.img_patches_list = np.asarray(self.img_patches_list)

    def __len__(self):
        return self.patches_num
        
    def __getitem__(self, index):
        img = self.img_patches_list[index]
        img = np.expand_dims(img, 0)
        img = torch.from_numpy(img)
        return img

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from ryu_pytools import arr_info, tensor_to_ndarr

    ds = skels_dataset_test('/home/ryuuyou/E5/project/data/skels_aug/img_deconv', 
                       patch_size=64, 
                       overlap=0)
    print(len(ds))
    
    # for i in range(2):
    #     img = ds.__getitem__(i*8)
    #     arr_info(img, 'img')
    #     tiff.imwrite(f'img{i}.tif', tensor_to_ndarr(img[0]))

    dl = DataLoader(ds, batch_size=100)
    for i, img in enumerate(dl):
        print(f'{i}: [{img.shape}]')
        arr_info(img, 'img')
