from torch import optim, nn
import torch


from dataset import MultiResDataset
from DataIterator import DataIterator


class Trainer:
    def __init__(
        self,
        G,
        D,
        default_G_lr,
        G_lr_dict,
        default_D_lr,
        D_lr_dict,
        default_minibatch,
        minibatch_dict,
        initial_llvl,
        max_llvl,
        D_opt_param,
        G_opt_param,
        stablize_imgs,
        fade_in_imgs,
        dataset_args,
        dataloader_args,
        dlatent_size,
        G_loss,
        D_loss
    ):
        assert type(G_lr_dict) is dict
        assert type(D_lr_dict) is dict
        assert type(minibatch_dict) is dict

        self.G = G
        self.D = D

        self.default_minibatch = default_minibatch
        self.default_D_lr = default_D_lr
        self.default_G_lr = default_G_lr
        self._G_lr_dict = G_lr_dict
        self._D_lr_dict = D_lr_dict
        self._minibatch_dict = minibatch_dict
        self.G_opt_Param = G_opt_param
        self.D_opt_param = D_opt_param
        self.G_loss = G_loss
        self.D_loss = D_loss
        self.dlatent_size = dlatent_size

        self.max_llvl = max_llvl
        self.stablize_imgs = stablize_imgs
        self.fade_in_imgs = fade_in_imgs
        self.imgs_per_layer = stablize_imgs + fade_in_imgs
        
        self.dataloader_args = dataloader_args
        self.dataset = MultiResDataset(
            dataset_args.root_dir,
            dataset_args.transforms,
            self.llvl)

        self.initial_llvl = initial_llvl
        self.llvl = initial_llvl
        self.nimg = 0

    def require_grad(self, model, value):
        assert type(value) is bool
        for param in model.parameters():
            param.requires_grad = value

    def train_G(self):
        self.G.zero_grad()
        self.require_grad(self.G, True)
        self.require_grad(self.D, False)
        self.G_loss(self)        

    def train_D(self):
        self.D.zero_grad()
        self.require_grad(self.G, False)
        self.require_grad(self.D, True)
        self.D_loss(self)      

    def initialize_opt(self):
        self.G_lr = self._G_lr_dict.get(self.resolution, self.default_G_lr)
        self.D_lr = self._D_lr_dict.get(self.resolution, self.default_D_lr)
        self.G_opt = optim.Adam(
            G.parameters(), lr=self.G_lr, **self.G_opt_param)
        self.D_opt = optim.Adam(
            D.parameters(), lr=self.D_lr, **self.D_opt_param)

    def update_batch_size(self):
        self.minibatch_size = self._minibatch_dict.get(
            self.resolution, self.default_minibatch)
        self.dataloader = DataIterator(
            self.dataset,
            self.minibatch_size,
            **self.dataloader_args)

    @property
    def llvl(self):
        return self._llvl

    @llvl.setter
    def llvl(self, value):
        assert type(value) is float
        assert 3 < value < self.max_llvl
        t = self._llvl
        self._llvl = self.max_llvl if value > self.max_llvl else value
        if int(t) < int(self._llvl):
            self.resolution = 2 ** int(self.llvl)

    @property
    def nimg(self):
        return self._nimg

    @nimg.setter
    def nimg(self, value):
        assert type(value) is int
        assert value > 0
        self._nimg = value
        self.llvl = self.initial_llvl + value / self.imgs_per_layer

    @property
    def resolution(self):
        return self._resolution
    
    @resolution.setter
    def resolution(self, value):
        assert type(value) is int
        assert value > 0
        self._resolution = value
        self.dataloader.dataset.resolution = self.resolution
        self.initialize_opt()
        self.update_batch_size()


