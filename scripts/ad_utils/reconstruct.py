from improved_diffusion import dist_util, logger
import functools
import torch as th
import numpy as np

def arbitrary_shot_reconstruction(
    model,
    diffusion,
    x_0,
    m_shot=1,
    k_step=50
):
    model.eval()
    with th.no_grad():
        t = th.from_numpy(np.repeat(k_step, x_0.shape[0])).long().to(dist_util.dev())
        epsilon = th.randn_like(x_0)

        x_t = diffusion.q_sample(x_0, t, epsilon)
        x_tilda_0 = model(x_t, diffusion._scale_timesteps(t))
        mse = th.mean((x_tilda_0 - x_0) ** 2, dim=(1, 2, 3))
        return x_tilda_0
    
