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
    
def normalize(x, eps=1e-8):
    return x * torch.rsqrt(torch.mean(x, dim=1, keepdim=True) + eps)

class apply_noise(nn.Module):
    def __init__(self, feat_size):
        # size: weight size = channel of feature map
        super().__init__()
        self.weight = torch.randn(feat_size, requires_grad=True)

    def forward(self, x, blocks):
        # make noise
        noise = torch.randn(size=(1, 1, 2**blocks, 2**blocks))
        # print(noise.size(),"= [1, 1, 8, 8]")
        # print(self.weight.size(),"= [8]")
        # TODO reshape noise
        # add noise
        x = x + noise*self.weight.reshape([1, -1, 1, 1])
        return x

class AdaIN(nn.Module):
    def __init__(self, dlatent_size, feat_size):
        super().__init__()
        self.dlatent_size = dlatent_size
        self.feat_size = feat_size
        self.transformation = nn.Linear(dlatent_size, 2*feat_size)

    def instance_norm(self, x, eps=1e-8):
        return (x-torch.mean(x, dim=[2,3], keepdim=True))*torch.rsqrt(torch.mean(x, dim=[2,3], keepdim=True) + eps)

    def style_mod(self, x, dlatent):
        # dlatent shape [4, 1, 512]
        # style shape [4, 2*512]
        style = self.transformation(dlatent)
        # style shape [b, 2, feat_size, 1, 1] [4, 2, 512, 512, 1, 1]
        style = style.reshape([-1, 2, self.feat_size] + [1]*2)
        # style 적용
        return x * (style[:,0]+1) + style[:,1]

    def forward(self, x, dlatent):
        x = self.instance_norm(x)
        x = self.style_mod(x, dlatent)
        return x
 

class block(nn.Module):
    def __init__(self, blocks=3, dlatent_size = 512):
        super().__init__()
        self.blocks = blocks
        self.dlatent_size = dlatent_size
        self.feat_size = min(int(8192/(2**blocks)), 512)
        self.apply_noise = apply_noise(self.feat_size)

        if blocks != 3:
            # self.conv1 = nn.Conv2d()
            pass
        else:
            self.const = torch.ones((1, dlatent_size, 2**blocks, 2**blocks), requires_grad=True)
        self.AdaIN = AdaIN(self.dlatent_size, self.feat_size)
        self.conv2 = nn.Conv2d(in_channels=dlatent_size, out_channels=dlatent_size, kernel_size=3, padding=1) 
        self.lrelu = nn.LeakyReLU(0.2)
        

    def forward(self, w, x = None):
        if self.blocks != 3 or x != None:
            x = self.conv1(x)
        else:
            x = self.const
        # print(x.size(), "= [1, 512, 8, 8]")
        x = self.apply_noise(x, blocks=self.blocks)
        x = normalize(self.lrelu(x))
        x = self.AdaIN(x, w)
        '''
        
        x = self.conv2(x)
        x = self.apply_noise()
        x = normalize(self.lrelu(x))
        x = self.style_mod2(self.instance_norm(x))
        '''
        return x

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

        self.block1 = block(blocks=3, dlatent_size=512)

    def forward(self, dlatents_in, layer_level, noise_in=None):
        '''
        Arguments
            dlatents_in: style vector w [b, #layers(14), 512]
            noise_in: noise list len: #layers
        '''
        images_out = 3
        # 8x8
        if layer_level == 3:
            # w [b, 14, 512]에서 [b, 2, 512]만 w로 넘겨줌
            images_out = self.block1(w=dlatents_in[:, 0])
            return images_out
        '''
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
        '''

        return images_out

