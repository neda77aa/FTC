import torch
import torch.nn.functional as F
import torch.nn as nn

#print(torch.__version__)

# KL divergence between the dirichlet distribution parameterized by alpha and the uniform dirichlet
# computes KLD per sample, expects input size of BxC
def KL(alpha, device):
    beta = torch.ones(1,alpha.size()[1]).to(device)
    S_alpha = torch.sum(alpha, 1, keepdim=True)
    S_beta = torch.sum(beta, 1, keepdim=True)
    #print(S_alpha)
    #print(S_beta)
    
    log_gamma_alpha = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), 1, keepdim=True)
    log_gamma_beta = torch.lgamma(S_beta) - torch.sum(torch.lgamma(beta), 1, keepdim=True)
    #print(log_gamma_alpha)
    #print(log_gamma_beta)
    
    dg_a = torch.digamma(alpha)
    dg_a0 = torch.digamma(S_alpha)
    #print(dg_a)
    #print(dg_a0)
    
    kl = torch.sum((alpha-beta)*(dg_a-dg_a0) , 1, keepdim=True) + log_gamma_alpha - log_gamma_beta
    return kl

# MSE loss for evidential classification DNN
# First finds the expected probabilities for each class given dirichlet param. by alpha
# Then computes MSE and KLD components of the loss, expects alpha=BxC, p=BxC
def evidential_mse(alpha, p, device, weights=None):
    S = torch.sum(alpha, 1, keepdim=True)
    evidence = alpha - 1
    predicted_mass = alpha/S
    sq_err = (p-predicted_mass)**2
    if weights is not None:
        weighted_sq_err = sq_err * weights
    else:
        weighted_sq_err = sq_err
    #print(weighted_sq_err)
    sq_loss = torch.sum(weighted_sq_err, 1, keepdim=True)
    #print(sq_loss)
    dirichlet_variance = alpha*(S-alpha)/(S*S*(S+1))
    var = torch.sum(dirichlet_variance, 1, keepdim=True)
    #print(var)
    alpha_tilde = evidence*(1-p) + 1
    #print(alpha_tilde)
    kl = KL(alpha_tilde, device)
    return (sq_loss + var), kl

def evidential_ce(alpha, p, device):
    S = torch.sum(alpha, 1, keepdim=True)
    evidence = alpha - 1
    predicted_mass = alpha/S
    ce_loss = F.binary_cross_entropy(predicted_mass, p)
    #print(sq_loss)
    dirichlet_variance = alpha*(S-alpha)/(S*S*(S+1))
    var = torch.sum(dirichlet_variance, 1, keepdim=True)
    #print(var)
    alpha_tilde = evidence*(1-p) + 1
    #print(alpha_tilde)
    kl = KL(alpha_tilde, device)
    return (ce_loss + var), kl

# wrapper function for evidential_mse
# expects pred=BxC
def evidential_loss(pred, target, nclasses):
    alpha = F.softplus(pred)+1
    target_oh = F.one_hot(target, nclasses).float()
    mse, kl = evidential_ce(alpha, target_oh, alpha.device)
    loss = torch.mean(mse + 0.1*kl)
    return loss

# get the probability and evidential vacuity based on the network's logits
def evidential_prob_vacuity(logits, nclasses):
    alpha = F.softplus(logits)+1
    S = torch.sum(alpha, dim=1, keepdim=True)
    prob = alpha / S
    vacuity = nclasses / S
    return prob, vacuity

# CDF based probability dist. approximation from Zhibin
# given parameters to the laplacian distribution mu/sigma, get the discrete probabilities
# for a distribution between 0 and 1, of n bins. Expects mu/sigma Bx2
def laplace_cdf(mu_sigma, num_classes, device):
    mu = mu_sigma[:,0:1]
    sigma = mu_sigma[:,1:2]
    uv = torch.linspace(0, 1, steps=num_classes+1).to(device)
    u = uv[1:]
    v = uv[:-1]
    # print(mu)
    # print(sigma)
    # print(u)
    # print(v)
    epsilon = torch.tensor(1e-8).to(device)
    p = 0.5 * ( torch.sign(u-mu)*(1 - torch.exp(-torch.abs(u-mu) / sigma)) - torch.sign(v-mu)*(1 - torch.exp(-torch.abs(v-mu) / sigma)) ) + epsilon
    p = p / torch.sum(p, 1, keepdim=True)
    return p

def laplace_cdf_loss(pred, target, nclasses):
    pred_categorical = laplace_cdf(F.sigmoid(pred), nclasses, pred.device)
    target_oh = 0.9*F.one_hot(target, nclasses) + 0.1/nclasses
    loss = F.binary_cross_entropy(pred_categorical, target_oh)
    return loss

class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss