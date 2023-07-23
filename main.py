import torch
from torch.utils.data import DataLoader


from mvtec_ad import MVTecAD
from improved_diffusion.unet import UNetModel





if __name__ == "__main__":
    # define transforms
    

    batch_size = 4
    num_workers = 8
    val_ratio = 0.1
    random_state = 42

    torch.manual_seed(random_state)

    # load data
    train_dataset = MVTecAD(
        "data",
        subset_name="bottle",
        train=True,
        transform=transform,
        mask_transform=transform,
        target_transform=target_transform,
        download=True,
    )

    test_dataset = MVTecAD(
        "data",
        subset_name="bottle",
        train=False,
        transform=transform,
        mask_transform=transform,
        target_transform=target_transform,
        download=True,
    )

    # feed to data loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )


    

    

    
