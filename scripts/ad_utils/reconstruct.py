from improved_diffusion import dist_util, logger
import functools

def arbitrary_shot_reconstruction(
    model,
    diffusion,
    schedule_sampler,
    image,
    m_shot=1
):
    t, weights = schedule_sampler.sample(image.shape[0], dist_util.dev())

    compute_losses = functools.partial(
        diffusion.training_losses,
        model,
        image,
        t
    )
    
