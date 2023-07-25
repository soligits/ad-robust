import os
import argparse

from torchvision.datasets.utils import download_and_extract_archive
import subprocess

data_dict = {
    'mvtec_anomaly_detection': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938113-1629952094/mvtec_anomaly_detection.tar.xz'
}
subset_dict = {
    'bottle': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937370-1629951468/bottle.tar.xz',
    'cable': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937413-1629951498/cable.tar.xz',
    'capsule': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937454-1629951595/capsule.tar.xz',
    'carpet': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937484-1629951672/carpet.tar.xz',
    'grid': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937487-1629951814/grid.tar.xz',
    'hazelnut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937545-1629951845/hazelnut.tar.xz',
    'leather': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937607-1629951964/leather.tar.xz',
    'metal_nut': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420937637-1629952063/metal_nut.tar.xz',
    'pill': 'https://www.mydrive.ch/shares/43421/11a215a5749fcfb75e331ddd5f8e43ee/download/420938129-1629953099/pill.tar.xz',
    'screw': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938130-1629953152/screw.tar.xz',
    'tile': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938133-1629953189/tile.tar.xz',
    'toothbrush': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938134-1629953256/toothbrush.tar.xz',
    'transistor': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938166-1629953277/transistor.tar.xz',
    'wood': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938383-1629953354/wood.tar.xz',
    'zipper': 'https://www.mydrive.ch/shares/38536/3830184030e49fe74747669442f0f282/download/420938385-1629953449/zipper.tar.xz'
}

dataset_name = next(iter(data_dict.keys()))
subset_names = list(subset_dict.keys())

def download_subset(dataset_path, subset_name):
    filename = subset_name + '.tar.xz'
    download_and_extract_archive(subset_dict[subset_name], dataset_path, filename=filename)
    os.remove(os.path.join(dataset_path, filename))
    
def preprocess_subset(subset_path):
    subprocess.run("chmod -R 755 " + subset_path, shell=True)
    test_path = os.path.join(subset_path, 'test')
    train_path = os.path.join(subset_path, 'train')
    gt_path = os.path.join(subset_path, 'ground_truth')

    if not os.path.exists(gt_path):
        return

    subprocess.run("rm -rf " + gt_path, shell=True)
    
    for path in [test_path, train_path]:
        normal_path = os.path.join(path, "good")
        anomaly_dirs = os.listdir(path)
        anomaly_dirs.remove("good")
        anomaly_paths = [os.path.join(path, anomaly_dir) for anomaly_dir in anomaly_dirs]

        # move all normal images to data_path and add 0_ prefix
        i = 0
        for filename in os.listdir(normal_path):
            os.rename(os.path.join(normal_path, filename), os.path.join(path, "0_" + "{:04d}".format(i) + ".png"))
            i += 1
        
        # move all anamoly images to data_path and add 1_ prefix
        for anamoly_path in anomaly_paths:
            for filename in os.listdir(anamoly_path):
                os.rename(os.path.join(anamoly_path, filename), os.path.join(path, "1_" + "{:04d}".format(i) + ".png"))
                i += 1

        for anomaly_path in anomaly_paths:
            os.rmdir(anomaly_path)
        os.rmdir(normal_path)

def download_and_preprocess_subset(dataset_path, subset_name):
    os.makedirs(dataset_path, exist_ok=True, mode=0o755)
    subset_dir = os.path.join(dataset_path, subset_name)
    if not os.path.exists(subset_dir):
        download_subset(dataset_path, subset_name)
    preprocess_subset(subset_dir)
        

def main():
    args = create_argparser().parse_args()

    os.makedirs(args.root, exist_ok=True, mode=0o755)
    dataset_path = os.path.join(args.root, dataset_name)

    # download data
    if args.subset_name == "all":
        for subset_name in subset_names:
            download_and_preprocess_subset(dataset_path, subset_name)
    elif args.subset_name in subset_names:
        download_and_preprocess_subset(dataset_path, args.subset_name)
    else:
        raise ValueError(f"Invalid subset_name: {args.subset_name}")

def create_argparser():
    defaults = dict(
        root=os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "data")),
        subset_name="all"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=defaults["root"])
    parser.add_argument("--subset_name", type=str, default=defaults["subset_name"])
    return parser

    

if __name__ == "__main__":
    main()