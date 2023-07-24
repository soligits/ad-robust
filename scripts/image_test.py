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
from improved_diffusion.resample import create_named_schedule_sampler

from ad_utils.ad_score import anomaly_score
from ad_utils.reconstruct import arbitrary_shot_reconstruction


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
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

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
        reconstructed_images = arbitrary_shot_reconstruction(model, diffusion, schedule_sampler, images, m_shot=args.m_shot)
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
    logger.log("testing complete")

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
        attack_strength=0.2,
        clip_denoised=True,
        use_ddim=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(
        dict(
            data_dir=os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "..",
                "data",
                "mvtec_anomaly_detection",
                "bottle",
                "train",
                "good",
            ),
            image_size=256,
            num_channels=64,
            num_heads=1,
            attention_resolutions="8",
            diffusion_steps=defaults.k_steps,
            noise_schedule="linear",
            batch_size=1,
        )
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser

if __name__ == "__main__":
    main()