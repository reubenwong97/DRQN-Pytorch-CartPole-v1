def soft_copy(target, source, tau):
    """
    Function performs exponential moving average copy.

    args:
    -----
        target (nn.Module): target network that will be a copy of source.
        source (nn.Module): the source network that serves as the copy.
        tau (float): controls the rate of copy.
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def freeze_params(model):
    """
    Freeze model params, dont want target encoder to accumulate gradients
    """
    for param in model.parameters():
        param.requires_grad = False


def hard_copy(target, source, freeze=True):
    """
    Do exact copy, freeze if boolean is passed
    """
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

    # if freeze is True, we freeze gradients on the target model after copy
    if freeze:
        freeze_params(target)
