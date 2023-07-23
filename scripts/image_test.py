"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os

import numpy as np
import torch as th
import torch.distributed as dist

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)

from ad_utils.ad_score import anomaly_score


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

def main():
    args = create_argparser().parse_args()

    dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(args.model_path, map_location="cpu")
    )
    model.to(dist_util.dev())
    model.eval()
    
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
    )

    logger.log("testing...")
    predicted_labels = []
    true_labels = []
    for i, (images, labels) in enumerate(data):
        epsilon = np.normal(0, 1, size=images.shape)   
        reconstructed_images = arbitrary_shot_reconstruction(model, images, args.recconstruction_type, args.attack_type, epsilon, y=labels)
        # TODO: add l2_pgd attack
        for image, reconstructed, label in zip(images, reconstructed_images, labels):
            if anomaly_score(image, reconstructed) > args.anomaly_threshold:
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
            true_labels.append(label)
    
    logger.log("saving results...")
    np.savez(
        os.path.join(args.output_dir, "image_test.npz"),
        predicted_labels=np.array(predicted_labels),
        true_labels=np.array(true_labels),
    )
            


    dist.barrier()
    logger.log("sampling complete")

def create_argparser():
    defaults = dict(
        data_dir="",
        model_path="",
        batch_size=1,
        model_path="",
        k_steps=50,
        m_shot=1,
        anomaly_threshold=0.5,
        attak_type='l2_pgd',
        clip_denoised=True,
        use_ddim=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()