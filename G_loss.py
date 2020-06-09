from torch.autograd import Variable, grad
from torch import nn
import torch

def logistic_nonsaturating(trainer):
    latent = torch.randn(trainer.minibatch_size,
                         trainer.dlatent_size, device='cuda')
    fake_out = trainer.D(trainer.G(latent, trainer.llvl), trainer.llvl)
    nn.Softplus(-fake_out).backward()

def logistic_saturating(trainer):
    latent = torch.randn(trainer.minibatch_size,
                         trainer.dlatent_size, device='cuda')
    fake_out = trainer.D(trainer.G(latent, trainer.llvl), trainer.llvl)
    (-nn.Softplus(fake_out)).backward()
