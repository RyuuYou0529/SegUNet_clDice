import torch
import numpy as np

import os

def arr_info(arr, name:str='Array'):
    assert type(arr) in [np.ndarray, torch.Tensor], 'Invalid array type.'
    print('==========')
    print(f'[{name}]:\n'
          f'shape: {arr.shape}\n'
          f'dtype: {arr.dtype}\n'
          f'max: {arr.max()}\n'
          f'min: {arr.min()}\n'
          f'mean: {arr.mean()}\n'
          f'std: {arr.std()}\n'
          f'sum: {arr.sum()}')
    if type(arr) == torch.Tensor:
        print(f'device: {arr.device}')
    print('==========')

def check_dir(path, *, mode:str='r'):
    if os.path.exists(path):
            print(f'the directory already exists: ["{path}"]')
    else:
        if mode == 'r':
            os.makedirs(path)
            print(f'the directory has been created: ["{path}"]')
        elif mode == 'a':
            os.mkdir(path)
            print(f'the directory has been created: ["{path}"]')
        else:
            print(f'the directory doesn\'t exist: ["{path}"]')

def tensor_to_ndarr(t: torch.Tensor):
    return t.detach().cpu().numpy()