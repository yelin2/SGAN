import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os

# transform에 resize 추가하지 마셈


class MultiResDataset(Dataset):
    def __init__(self, root_dir, transform=None, llvl=8):
        # layer level
        self.root_dir = root_dir
        self.transform = transform
        self.llvl = llvl

    def __len__(self):
        return 70000

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
    def llvl(self):
        return self._llvl
    
    @llvl.setter
    def llvl(self, value):
        assert 0 < value
        self._llvl = value
        self.resolution = 2 ** self._llvl
        self.transform_resize = transforms.Resize((self.resolution, self.resolution))


