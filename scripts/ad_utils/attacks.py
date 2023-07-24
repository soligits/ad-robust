from .reconstruct import arbitrary_shot_reconstruction
from torch.linalg import vector_norm
import torch

def projection(x1, x2, epsilon, ord=2):
    diff = x1 - x2
    diff = diff / vector_norm(diff, ord) * epsilon
    return x2 + diff


def pgd_attack(x, y, model, diffusion, ad_score, m_shot=1, k_step=50, norm='l2', epsilon=0.2, alpha=1e-3, N=100):
    adversary = x
    y = -1 if y == 1 else 1
    n = 0
    while n < N:
        reconstruction = arbitrary_shot_reconstruction(model, diffusion, adversary, m_shot=m_shot, k_step=k_step)
        score = ad_score(x, reconstruction)
        score.backward()
        new_adversary = None
        ord = 2 if norm == 'l2' else float('inf') 
        new_adversary = None
        if ord == 2:
            new_adversary = adversary + alpha * y * adversary / vector_norm(adversary)
        else:
            new_adversary = adversary + alpha * torch.sgn(y * adversary.grad)
        new_adversary = projection(adversary, new_adversary, epsilon, ord)
        adversary = new_adversary
        n += 1
        
    return adversary, arbitrary_shot_reconstruction(model, diffusion, adversary, m_shot=m_shot, k_step=k_step)