import torch
import torch.nn.functional as F
import numpy as np


def onehot_from_logits(logits, eps=0.01, device='cpu'):
    """
    Given batch of logits, return one-hot sample using epsilon greedy strategy
    (based on given epsilon)
    """
    # get best (according to current policy) actions in one-hot form
    argmax_acs = (logits == logits.max(1, keepdim=True)[0]).float().to(device=device)
    return argmax_acs
    # # get random actions in one-hot form
    # eye = torch.eye(logits.shape[1])
    # rand_choice = eye[
    #             [np.random.choice(range(logits.shape[1]), size=logits.shape[0])],
    #     ]
    # rand_acs = torch.tensor(rand_choice, requires_grad=False
    # ).to(device=device)

    # # chooses between best and random actions using epsilon greedy
    # return torch.stack(
    #     [argmax_acs[i] if r > eps else rand_acs[i]
    #      for i, r in enumerate(torch.rand(logits.shape[0]))]
    # )


def sample_gumbel(shape, eps=1e-20, tens_type=torch.FloatTensor, device='cpu'):
    """Sample from Gumbel(0, 1)"""
    U = torch.tensor(tens_type(*shape).uniform_(),
                 requires_grad=False).to(device=device)
    return -torch.log(-torch.log(U + eps) + eps)


def gumbel_softmax(logits, temperature=1.0, device='cpu'):
    """Sample from the Gumbel-Softmax distribution and optionally discretize.
    Args:
      logits: [batch_size, n_class] unnormalized log-probs
      temperature: non-negative scalar
      hard: if True, take argmax, but differentiate w.r.t. soft sample y
    Returns:
      [batch_size, n_class] sample from the Gumbel-Softmax distribution.
      If hard=True, then the returned sample will be one-hot, otherwise it will
      be a probabilitiy distribution that sums to 1 across classes
    """
    y = gumbel_softmax_sample(logits, temperature, device=device)
    y_hard = onehot_from_logits(y, device=device)
    y = (y_hard - y).detach() + y
    return y


def gumbel_softmax_sample(logits, temperature, device='cpu'):
    """ Draw a sample from the Gumbel-Softmax distribution"""
    y = logits + sample_gumbel(logits.shape,
                               tens_type=type(logits.data), device=device)
    return F.softmax(y / temperature, dim=1)
