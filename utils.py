from torchvision import transforms

def _convert_label(x):
    """
    convert anomaly label. 0: normal; 1: anomaly.
    :param x (int): class label
    :return: 0 or 1
    """
    return 0 if x == 0 else 1

def create_input_transform():
    transform = transforms.Compose(
        [transforms.Resize((256, 256)), transforms.ToTensor()]
    )
    return transform

def create_target_transform():
    target_transform = transforms.Lambda(_convert_label)
    return target_transform
