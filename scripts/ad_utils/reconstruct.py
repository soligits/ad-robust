from improved_diffusion import dist_util, logger
import functools
import torch as th
import numpy as np

def _reconstruct(
    model,
    diffusion,
    x_0,
    m_shot=1,
    k_step=50,
):
    t = th.from_numpy(np.repeat(k_step, x_0.shape[0])).long().to(dist_util.dev())
    epsilon = th.randn_like(x_0)

    x_t = diffusion.q_sample(x_0, t, epsilon)
    s = th.from_numpy(np.repeat(k_step - m_shot + 1, x_0.shape[0])).long().to(dist_util.dev())
    x_tilda_0 = model(x_t, diffusion._scale_timesteps(s))
    return x_tilda_0

def arbitrary_shot_reconstruction(
    model,
    diffusion,
    x_0,
    m_shot=1,
    k_step=50,
    requires_grad=False
):
    model.train()
    if requires_grad:
        return _reconstruct(model, diffusion, x_0, m_shot, k_step)
    else:
        with th.no_grad():
            return _reconstruct(model, diffusion, x_0, m_shot, k_step)

