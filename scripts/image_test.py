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
import torch

from ad_utils.ad_score import anomaly_score, calculate_err_ms
from ad_utils.reconstruct import arbitrary_shot_reconstruction
from ad_utils.metrics import AUPR, AUROC
from ad_utils.nn import ADScore
from ad_utils.attacks import pgd_attack


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

    logger.log("testing...")
    logger.log("Calculating multiscale error on training data...")

    training_err_ms = np.zeros((args.image_size, args.image_size))
    train_data = load_data(
        data_dir=args.train_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=False,
        deterministic=True,
    )
    train_len = len(os.listdir(args.train_dir))
    for i, (images, _) in enumerate(train_data):
        if i >= train_len / args.batch_size:
            break
        images = images.to(dist_util.dev())
        reconstructed_images = arbitrary_shot_reconstruction(
            model,
            diffusion,
            images,
            k_step=args.k_steps,
            m_shot=args.m_shot,
        )
        for image, reconstructed in zip(images, reconstructed_images):
            image = image.permute(1, 2, 0)
            reconstructed = reconstructed.permute(1, 2, 0)
            image = image.cpu().detach().numpy()
            reconstructed = reconstructed.cpu().detach().numpy()
            training_err_ms += calculate_err_ms(image, reconstructed)
    training_err_ms /= train_len

    logger.log("Finished calculating multiscale error on training data...")
    logger.log("Calculating standard AUC and AUPR on test data...")

    test_data = load_data(
        data_dir=args.test_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        deterministic=True,
    )
    test_len = len(os.listdir(args.test_dir))
    predicted_labels = []
    true_labels = []
    for i, (images, labels) in enumerate(test_data):
        if i >= test_len / args.batch_size:
            break
        labels = labels["y"]
        images = images.to(dist_util.dev())
        labels = labels.to(dist_util.dev())
        reconstructed_images = arbitrary_shot_reconstruction(
            model,
            diffusion,
            images,
            k_step=args.k_steps,
            m_shot=args.m_shot,
        )
        for image, reconstructed, label in zip(images, reconstructed_images, labels):
            image = image.permute(1, 2, 0)
            reconstructed = reconstructed.permute(1, 2, 0)
            image = image.cpu().detach().numpy()
            reconstructed = reconstructed.cpu().detach().numpy()
            if (
                anomaly_score(image, reconstructed, training_err_ms)
                > args.anomaly_threshold
            ):
                predicted_labels.append(1)
            else:
                predicted_labels.append(0)
            true_labels.append(label.cpu().detach().item())

    aupr = AUPR(true_labels, predicted_labels)
    auroc = AUROC(true_labels, predicted_labels)

    logger.log(f"AUPR: {aupr}")
    logger.log(f"AUROC: {auroc}")
    logger.log("Finish calculating standard AUC and AUPR on test data...")
    logger.log("Calculating robust AUC and AUPR on test data...")

    pgd_l2_labels = []
    pgd_l_inf_labels = []
    ad_score = ADScore(
        mean_filter=(torch.ones((1, 1, 3, 3)) / 3 * 3).to(dist_util.dev())
    )
    training_err_ms = torch.Tensor(training_err_ms).to(dist_util.dev())
    for i, (images, labels) in enumerate(test_data):
        if i >= test_len / args.batch_size:
            break
        labels = labels["y"]
        images = images.to(dist_util.dev())
        labels = labels.to(dist_util.dev())
        reconstructed_images = arbitrary_shot_reconstruction(
            model,
            diffusion,
            images,
            k_step=args.k_steps,
            m_shot=args.m_shot,
        )
        # TODO: add l2_pgd attack
        for image, reconstructed, label in zip(images, reconstructed_images, labels):
            adv_l2, adv_recon_l2 = pgd_attack(
                image,
                label,
                training_err_ms,
                model,
                diffusion,
                ad_score,
                m_shot=args.m_shot,
                k_step=args.k_steps,
                N=10,
                alpha=1e-2,
                norm='l2'
            )
            adv_l_inf, adv_recon_l_inf = pgd_attack(
                image,
                label,
                training_err_ms,
                model,
                diffusion,
                ad_score,
                m_shot=args.m_shot,
                k_step=args.k_steps,
                N=10,
                alpha=1e-2,
                norm='l_inf'
            )
            if (
                ad_score(adv_l2, adv_recon_l2, training_err_ms).cpu().item()
                > args.anomaly_threshold
            ):
                pgd_l2_labels.append(1)
            else:
                pgd_l2_labels.append(0)
            if (
                ad_score(adv_l_inf, adv_recon_l_inf, training_err_ms).cpu().item()
                > args.anomaly_threshold
            ):
                pgd_l_inf_labels.append(1)
            else:
                pgd_l_inf_labels.append(0)

    aupr_l2 = AUPR(true_labels, pgd_l2_labels)
    auroc_l2 = AUROC(true_labels, pgd_l2_labels)
    aupr_l_inf = AUPR(true_labels, pgd_l_inf_labels)
    auroc_l_inf = AUROC(true_labels, pgd_l_inf_labels)
    logger.log(f"AUPR PGD_L2: {aupr_l2}")
    logger.log(f"AUROC PGD_L2: {auroc_l2}")
    logger.log(f"AUPR PGD_L_INF: {aupr_l_inf}")
    logger.log(f"AUROC PGD_L_INF: {auroc_l_inf}")

    logger.log("Finish calculating robust AUC and AUPR on test data...")

    dist.barrier()
    logger.log("testing complete")


def create_argparser():
    defaults = dict(
        train_dir="/home/soltani/Workspace/ad-robust/data/mvtec_anomaly_detection/bottle/train/good",
        test_dir="/home/soltani/Workspace/ad-robust/data/mvtec_anomaly_detection/bottle/test",
        batch_size=1,
        model_path="/home/soltani/Workspace/ad-robust/blob/ema_0.9999_003000.pt",
        k_steps=50,
        m_shot=1,
        anomaly_threshold=0.5,
        attak_type="l2_pgd",
        attack_strength=0.2,
        clip_denoised=True,
        use_ddim=False,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(
        dict(
            image_size=256,
            num_channels=64,
            num_heads=1,
            attention_resolutions="8",
            diffusion_steps=1000,
            noise_schedule="linear",
            batch_size=1,
        )
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
