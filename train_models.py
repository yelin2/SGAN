import torch.nn as nn
import torch
import torchvision
import numpy as np
import time

from models.style_G import G_mapping, G_synthesis

# TODO study multi-gpu 
def main():
    G = G_mapping()
    G_syn = G_synthesis(dlatent_size=512, num_channels=3, resolution=512, blur_filter=[1,2,1])
    # G_parallel = torch.nn.DataParallel(G, device_ids=[0, 1, 2, 3]).to('cuda')

    latent = torch.randn((4, 512))
    # res = G_parallel(torch.from_numpy(latent).float())
    res = G(latent.float())
    res1 = G_syn(res, layer_level=9)
    print(res.shape)
    print(res1.shape)

if __name__ == '__main__':
    main()