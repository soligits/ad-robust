from .reconstruct import arbitrary_shot_reconstruction
from torch.linalg import vector_norm
import torch


def projection(x1, x2, epsilon, ord=2):
    diff = x1 - x2
    diff = diff / vector_norm(diff, ord) * epsilon
    return x2 + diff


def pgd_attack(
    x,
    y,
    t,
    model,
    diffusion,
    ad_score,
    m_shot=1,
    k_step=50,
    norm="l2",
    epsilon=0.2,
    alpha=1e-3,
    N=100,
):
    adversary = x
    t.requires_grad = True
    adversary.requires_grad = True
    model.train()
    y = -1 if y == 1 else 1
    n = 0
    while n < N:
        print(n)
        reconstruction = arbitrary_shot_reconstruction(
            model, diffusion, adversary.unsqueeze(dim=0), m_shot=m_shot, k_step=k_step, requires_grad=True
        ).squeeze(dim=0)
        assert adversary.shape == reconstruction.shape
        score = ad_score(adversary, reconstruction, t)
        score.backward()
        new_adversary = None
        ord = 2 if norm == "l2" else float("inf")
        if ord == 2:
            new_adversary = adversary + alpha * y * adversary.grad / vector_norm(adversary.grad)
        else:
            new_adversary = adversary + alpha * torch.sgn(y * adversary.grad)
        new_adversary = projection(adversary.detach(), new_adversary.detach(), epsilon, ord)
        adversary = new_adversary
        adversary.requires_grad = True
        n += 1

    adversary = adversary.detach()
    assert adversary.shape == x.shape
    return adversary, arbitrary_shot_reconstruction(
        model, diffusion, adversary.unsqueeze(dim=0), m_shot=m_shot, k_step=k_step
    ).squeeze(dim=0)
