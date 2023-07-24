
def arbitrary_shot_reconstruction(model, image, reconstruction_type, attack_type, epsilon, **model_kwargs):
    if reconstruction_type == 'one_shot':
        reconstructed_images = diffusion.p_sample_loop(
            model,
            image,
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
        )
    elif reconstruction_type == 'iterative':
        reconstructed_images = iterative_reconstruction(model, image, attack_type, epsilon, **model_kwargs)
    return reconstructed_images