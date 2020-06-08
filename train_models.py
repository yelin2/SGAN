import torch.nn as nn
import torch
import torchvision
import numpy as np
import time

from models.style_G import G_mapping

# TODO study multi-gpu 
def main():
    curr = time.time()
    G = G_mapping()
    G_parallel = torch.nn.DataParallel(G, device_ids=[0, 1, 2, 3]).to('cuda')
    
    for i in range(100):
        curr = time.time()
        latent = np.random.randn(4, 512)
        res = G_parallel(torch.from_numpy(latent).float())
        print(res.shape)
        print(time.time()-curr)
        

if __name__ == '__main__':
    main()