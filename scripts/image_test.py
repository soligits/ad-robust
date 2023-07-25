"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
import functools
import argparse
import os

import matplotlib.pyplot as plt

import numpy as np
import torch as th
import torch.distributed as dist
import torch

from sklearn.metrics import roc_auc_score

from improved_diffusion import dist_util, logger
from improved_diffusion.image_datasets import load_data
from improved_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torch

from ad_utils.reconstruct import arbitrary_shot_reconstruction
from ad_utils.nn import ADScore
from ad_utils.attacks import pgd_attack


def calculate_train_err_ms(args, model, diffusion, ad_score):
    train_data = load_data(
        data_dir=args.train_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=False,
        deterministic=True,
    )
    train_len = len(os.listdir(args.train_path))
    training_err_ms = torch.zeros((args.image_size, args.image_size)).to(
        dist_util.dev()
    )

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
            training_err_ms += ad_score._calculate_err_ms(image, reconstructed)
    training_err_ms /= train_len

    return training_err_ms


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

    s = args.mean_filter_size
    ad_score = ADScore(
        mean_filter=(torch.ones((1, 1, s, s)) / s * s).to(dist_util.dev())
    )

    logger.log("testing...")

    logger.log("Calculating multiscale error on train data...")
    training_err_ms = calculate_train_err_ms(args, model, diffusion, ad_score)
    logger.log("Finished calculating multiscale error on train data...")

    logger.log("Calculating standard AUC on test data...")

    test_data = load_data(
        data_dir=args.test_path,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        deterministic=True,
    )
    test_len = len(os.listdir(args.test_path))

    reconstruct = lambda x_0: arbitrary_shot_reconstruction(
            model,
            diffusion,
            x_0,
            k_step=args.k_steps,
            m_shot=args.m_shot,
            requires_grad=False,
    )
        
    true_labels = []
    predicted_scores = []
    for i, (images, labels) in enumerate(test_data):
        if i >= test_len / args.batch_size:
            break
        labels = labels["y"]
        images = images.to(dist_util.dev())
        labels = labels.to(dist_util.dev())
        reconstructed_images = reconstruct(images)
        for image, reconstructed, label in zip(images, reconstructed_images, labels):
            score = ad_score(image, reconstructed, training_err_ms).cpu().detach().item()
            true_labels.append(label.cpu().detach().item())
            predicted_scores.append(score)

    auc = roc_auc_score(true_labels, predicted_scores)
    logger.log(f"AUC: {auc}")
    logger.log("Finish calculating standard AUC on test data...")


    logger.log("Calculating robust AUC on test data...")
    attack = lambda x, y: pgd_attack(
                x,
                y,
                training_err_ms,
                model,
                diffusion,
                ad_score,
                m_shot=args.m_shot,
                k_step=args.k_steps,
                epsilon=args.attack_strength,
                N=args.attack_n,
                alpha=args.attack_alpha,
                norm= "l2" if args.attack_type == "l2_pgd" else "l_inf",
            )

    pgd_predicted_scores = []
    for i, (images, labels) in enumerate(test_data):
        if i >= test_len / args.batch_size:
            break
        labels = labels["y"]
        images = images.to(dist_util.dev())
        labels = labels.to(dist_util.dev())
        reconstructed_images = reconstruct(images)
        for image, reconstructed, label in zip(images, reconstructed_images, labels):
            adv, adv_recon = attack(image, label)
            score = ad_score(adv, adv_recon, training_err_ms).cpu().item()
            pgd_predicted_scores.append(score)


    pgd_auc = roc_auc_score(true_labels, pgd_predicted_scores)
    logger.log(f"AUC {args.attack_type}: {pgd_auc}")

    logger.log("Finish calculating robust AUC on test data...")

    dist.barrier()
    logger.log("testing complete")


def create_argparser():
    defaults = dict(
        train_path="/home/soltani/Workspace/ad-robust/data/mvtec_anomaly_detection/bottle/train",
        test_path="/home/soltani/Workspace/ad-robust/data/mvtec_anomaly_detection/bottle/test",
        batch_size=1,
        model_path="/home/soltani/Workspace/ad-robust/blob/ema_0.9999_003000.pt",
        k_steps=50,
        m_shot=1,
        mean_filter_size=3,
        anomaly_threshold=0.5,
        attack_type="l2_pgd",
        attack_strength=0.2,
        attack_n=10,
        attack_alpha=1e-2,
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
