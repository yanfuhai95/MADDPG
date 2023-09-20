import torch
import torch.nn.functional as F
import numpy as np


def onehot_from_logits(logits, eps=0.1, device='cpu'):
    argmax_acs = (logits == logits.max(1, keepdim=True)
                  [0]).float().to(device=device)
    if eps == 0.0:
        return argmax_acs

    rand_acs = torch.tensor(
        torch.eye(logits.shape[1])[
            [np.random.choice(range(logits.shape[1]), size=logits.shape[0])]
        ],
        device=device,
        requires_grad=False)

    return torch.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(torch.rand(logits.shape[0]))])


def sample_gumbel(shape):
    return -torch.log(-torch.log(torch.rand(shape)))


def gumbel_softmax_sample(logits, temperature):
    y = logits + sample_gumbel(logits.shape).to(logits.device)
    return F.softmax(y / temperature, dim=1)


def gumbel_softmax(logits, temperature=1.0, eps=0.01):
    y = gumbel_softmax_sample(logits, temperature)
    y_hard = onehot_from_logits(y, eps=eps)
    y = (y_hard.to(logits.device) - y).detach() + y
    return y
