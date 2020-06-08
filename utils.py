def set_lr(optim, lr):
    for param_group in optim.pram_groups:
        param_group['lr'] = lr

