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


def sample_gumbel(shape, device='cpu'):
    return -torch.log(-torch.log(torch.rand(shape).to(device=device)))


def gumbel_softmax(logits, device='cpu'):
    y = logits + sample_gumbel(logits.shape, device=device)
    return F.softmax(y, dim=-1)
