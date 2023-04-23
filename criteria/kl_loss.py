import torch
from torch import nn
from torch import distributions

class NormalKLLoss(nn.Module):
    """
    NormalKLLoss
    """
    def __init__(self, reduction='mean'):
        super(NormalKLLoss, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        self.reduction = reduction

    def forward(self, q_mu, q_logvar, p_mu=None, p_logvar=None):
        """
        q_mu: (batch_size, latent_size)
        q_logvar: (batch_size, latent_size)
        """
        if p_mu is None:
            p_mu = torch.zeros_like(q_mu)
        if p_logvar is None:
            p_logvar = torch.zeros_like(q_logvar)

        q_norm = distributions.Normal(q_mu, q_logvar.exp().sqrt())
        p_norm = distributions.Normal(p_mu, p_logvar.exp().sqrt())
        kl = distributions.kl_divergence(q_norm, p_norm).sum(dim=1)

        if self.reduction == 'mean':
            kl = kl.mean()
        elif self.reduction == 'sum':
            kl = kl.sum()
        return kl