import torch.nn as nn
import torch
import torchvision
import numpy as np

from models.style_G import G_mapping

# convert to GPU operation
def main():
    G = G_mapping()
    latent = np.random.randn(4, 512)
    img = np.random.randn(4, 3, 64, 64)
    res = G(torch.from_numpy(latent).float())
    print(res.shape)

if __name__ == '__main__':
    main()