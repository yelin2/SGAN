import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# transform에 resize 추가하지 마셈


class MultiResDataset(Dataset):
    def __init__(self, root_dir, transform=None, resolution=8):
        # layer level
        self.root_dir = root_dir
        self.transform = transform
        self.resolution = resolution

    def __len__(self): 
        return 70000 #! 좀 더 사람답게 바꾸자 ^^

    def __getitem__(self, index):
        low_num = str(index % 1000).zfill(3)
        high_num = str(int(index / 1000)).zfill(2)
        img_path = self.root_dir + "/" + high_num + "000/" + high_num + low_num + ".png"
        img = Image.open(img_path)
        img = self.transform_resize(img)
        if self.transform:
            img = self.transform(img)
        
        return TF.to_tensor(img)

    @property
    def resolution(self):
        return self._resolution
    
    @resolution.setter
    def resolution(self, value):
        assert 0 < value
        self._resolution = value
        self.transform_resize = transforms.Resize((self.resolution, self.resolution))


