from torch.autograd import Variable, grad
from torch import nn
import torch


def logistic(trainer):
    latent = torch.randn(trainer.minibatch_size,
                         trainer.dlatent_size, device='cuda')
    fake_out = trainer.D(trainer.G(latent, trainer.llvl), trainer.llvl)
    real_out = trainer.D(next(trainer.dataloader), trainer.llvl)
    loss = nn.Softplus(fake_out)
    loss += nn.Softplus(-real_out)
    loss.backward()


def logistic_gp(trainer, r1_gamma=10.0, r2_gamma=0.0):
    latent = torch.randn(trainer.minibatch_size,
                         trainer.dlatent_size, device='cuda')
    fake_out = trainer.D(trainer.G(latent, trainer.llvl), trainer.llvl)
    fake_predict = nn.Sigmoid(fake_out).mean()
    fake_predict.backward()

    real_img = next(trainer.dataloader)
    real_out = trainer.D(real_img, trainer.llvl)
    #! TODO

    epsilon = torch.rand((fake_out.shape[0], 1, 1, 1)).cuda()
    x_hat = epsilon * real_img.data + (1 - epsilon) * fake_out.data
    x_hat.requires_grad = True
    hat_predict = trainer.D(x_hat, trainer.llvl)
    grad_x_hat = grad(outputs=hat_predict.sum(),
                      inputs=x_hat, create_graph=True)[0]
    gp = ((grad_x_hat.view(grad_x_hat.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()
    gp = 10 * gp
    gp.backward()
    
