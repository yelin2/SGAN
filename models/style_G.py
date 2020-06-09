import torch.nn as nn
import torch
import torchvision
import numpy as np

# TODO initialization
class G_mapping(nn.Module):
    def __init__(self, latents_size=512, labels_size=1, dlatent_size=512, dlatent_broadcast=14):
        '''
        Arguments:
            latents_size: input latent size [b, 512]
            labels_size: input label size? maybe None
            dlatent_size: output latent size [b, 512]
            dlatent_broadcast: number of layers to broadcast = number of G_synthesis layer(2 layer per block).
        '''

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
        
        # change torch.unsqueeze
        w = w[:,np.newaxis].repeat(1, self.dlatent_broadcast, 1)

        return w
    
class extract_style(nn.Module):
    def __init__(self, dlatent_size=512, feat_c=3):
        super.__init__()
        self.transformation = nn.Linear(dlatent_size, 2*feat_c)
    def forward(self, x):      
        return 1

class noise(nn.Module):
    def __init__(self):
        super.__init__()
    def forward(self):
        return 1

class AdaIN(nn.Module):
    def __init__(self):
        super.__init__()
    def forward(self):
        return 1

class block(nn.Module):
    def __init__(self):
        super.__init__()
    def forward(self):
        return 1

class G_synthesis(nn.Module):
    def __init__(self, dlatent_size=512, num_channels=3, resolution=512, blur_filter=[1,2,1]):
        '''
        Arguments:        
            dlatent_size: Disentangled latent (W) dimensionality. default 512
            num_channels: Number of output color channels. default 3
            resolution: Output resolution. default 512
            blur_filter: Low-pass filter to apply when resampling activations. default [1,2,1] None = no filtering. 

        fmap_base           = 8192,         # Overall multiplier for the number of feature maps.
        fmap_decay          = 1.0,          # log2 feature map reduction when doubling the resolution.
        fmap_max            = 512,          # Maximum number of feature maps in any layer.
        fused_scale         = 'auto',       # True = fused convolution + scaling, False = separate ops, 'auto' = decide automatically.
        is_template_graph   = False,        # True = template graph constructed by the Network class, False = actual evaluation.
        force_clean_graph   = False,        # True = construct a clean graph that looks nice in TensorBoard, False = default behavior.
        
        dtype = fp32, use instance norm, use pixel norm, use_wscale?, use leakyrelu, noise 매번 만들기, use noise , learning constant, use styles
        '''
        super().__init__()
        self.dlatent_size = dlatent_size
        self.num_channels = num_channels
        self.resolution = resolution
        self.blur_filter = blur_filter





    def forward(self, dlatents_in, noise_in, layer_level):
        '''
        Arguments
            dlatents_in: style vector w [b, #layers(14), 512]
            noise_in: noise list len: #layers
        '''
        images_out = 3
        # 8x8
            # do first block 
        if layer_level == 3:
            return images_out
        
        # 16x16
            # do block
        if layer_level == 4:
            # fade_in
            return images_out

        # 32x32
        if layer_level == 5:
            # fade_in
            return images_out
        
        # 64x64
        if layer_level == 6:
            # fade_in
            return images_out

        # 128x128
        if layer_level == 7:
            # fade_in
            return images_out

        # 256x256
        if layer_level == 8:
            # fade_in
            return images_out

        # 512x512
        if layer_level == 9:
            # fade_in
            return images_out



