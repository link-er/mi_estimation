import numpy as np
import torch
import torch.nn.functional as F

def logmeanexp_nodiag(f, device='cuda'):
    n = f.size(0)
    dim = (0, 1)

    # get log of sum of exponents of non-diagonal elements
    # essentially second summand of DV bound
    logsumexp = torch.logsumexp(
        f - torch.diag(np.inf * torch.ones(n).to(device)), dim=dim)

    try:
        #if len(dim) == 1:
        #    num_elem = batch_size - 1.
        #else:
        num_elem = n * (n - 1.)
    except ValueError:
        num_elem = n - 1
    # just need to normalize expectation, so sum is divided by the amount of elements
    return logsumexp - torch.log(torch.tensor(num_elem)).to(device)

def js_fgan_lower_bound(f, device='cuda'):
    """Lower bound on Jensen-Shannon divergence from Nowozin et al. (2016)."""
    f_diag = f.diag()
    first_term = np.log(2) - F.softplus(-f_diag).mean()

    n = f.size(0)
    f_others = f - torch.diag(np.inf * torch.ones(n)).to(device)
    second_term = - torch.log(2 - torch.exp(np.log(2) - (F.softplus(-f_others).sum()))) / (n * (n - 1.))

    return first_term + second_term

def estimate_mutual_information(x, y, critic_fn, clip=None):
    """Estimate variational lower bounds on mutual information.

  Args:
    x: [batch_size, dim_x] Tensor
    y: [batch_size, dim_y] Tensor
    critic_fn: callable that takes x and y as input and outputs critic scores
      output shape is a [batch_size, batch_size] matrix

  Returns:
    scalar estimate of mutual information
    """
    x, y = x.cuda(), y.cuda()
    f = critic_fn(x, y)

    if clip is not None:
        f_ = torch.clamp(f, -clip, clip)
    else:
        f_ = f
    z = logmeanexp_nodiag(f_)
    # DV bound: expectation of the f for joint distribution minus log of expectation of exponents of not joint distribution
    dv = f.diag().mean() - z

    # one more another lower bound on MI
    js = js_fgan_lower_bound(f)

    #TODO: we propagate gradients only through js? allowing dv seems to blow up the training to infinity
    #TODO: only js at the same moment converges from negative number to 0
    with torch.no_grad():
        dv_js = dv - js

    return js + dv_js