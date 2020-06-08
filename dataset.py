from torch.utils.data import Dataset
from torchvision import transforms
from skimage import io
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
        img = io.imread(img_path)
        if self.transform:
            img = transforms.Resize((self.resolution, self.resolution))
            img = self.transform(img)
        return img
