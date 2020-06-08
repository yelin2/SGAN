import torch.nn as nn
import torch
import torchvision
import numpy as np

# TODO
# initialization
class G_mapping(nn.Module):
    def __init__(self, latents_size=512, labels_size=1, dlatent_size=512, dlatent_broadcast=14):
        # latents_size: input latent size [b, 512]
        # labels_size: input label size? maybe None
        # dlatent_size: output latent size [b, 512]
        # dlatent_broadcast: number of layers to broadcast = number of G_synthesis layer(2 layer per block).
        super().__init__()

        self.latents_size = latents_size
        self.labels_size = labels_size
        self.dlatent_size = dlatent_size
        self.dlatent_broadcast = dlatent_broadcast

        self.labels_fc = nn.Linear(labels_size, latents_size)
        self.g_mapping = nn.Sequential(nn.Linear(latents_size, latents_size), nn.LeakyReLU(0.2), 
                                       nn.Linear(latents_size, latents_size), nn.LeakyReLU(0.2),
                                       nn.Linear(latents_size, latents_size), nn.LeakyReLU(0.2),
                                       nn.Linear(latents_size, latents_size), nn.LeakyReLU(0.2),
                                       nn.Linear(latents_size, latents_size), nn.LeakyReLU(0.2),
                                       nn.Linear(latents_size, latents_size), nn.LeakyReLU(0.2),
                                       nn.Linear(latents_size, latents_size), nn.LeakyReLU(0.2),
                                       nn.Linear(latents_size, dlatent_size), nn.LeakyReLU(0.2))
    

    def forward(self, x, labels=None):
        
        if labels is not None:
           y = self.labels_fc(labels)
           x = torch.cat(y,x)

        # pixel normalization
        epsilon = 1e-8
        x = x * torch.rsqrt(torch.mean(x, dim=1, keepdim=True) + epsilon)

        # forward 8 fc layer w [b, 512]
        w = self.g_mapping(x)

        # broadcasting w [b, dlatent_broadcast, 512]
            # repeat: copy tensor data
            # expand: view tensor data
        w = w[:,np.newaxis].repeat(1, self.dlatent_broadcast, 1)

        return w
    


