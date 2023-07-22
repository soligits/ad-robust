import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from mvtec_ad import MVTecAD


def _convert_label(x):
    """
    convert anomaly label. 0: normal; 1: anomaly.
    :param x (int): class label
    :return: 0 or 1
    """
    return 0 if x == 0 else 1


if __name__ == "__main__":
    # define transforms
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    target_transform = transforms.Lambda(_convert_label)

    batch_size = 2
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

    # check data
    for x, y, mask in train_loader:
        print(x.shape, y.shape, mask.shape)
        break
    
    for x, y, mask in test_loader:
        print(x.shape, y.shape, mask.shape)
        break
    
