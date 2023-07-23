import os
import argparse
from ad_utils.mvtec_ad import MVTecAD

def preprocess_test_data(root, subset_name):
    data_path = os.path.join(root, subset_name, "test")
    normal_path = os.path.join(data_path, "good")
    anamoly_paths = [
        os.path.join(data_path, "broken_large"),
        os.path.join(data_path, "broken_small"),
        os.path.join(data_path, "contamination")
    ]

    # move all normal images to data_path and add 0_ prefix

    i = 0
    for filename in os.listdir(normal_path):
        os.rename(os.path.join(normal_path, filename), os.path.join(data_path, "0_" + "{:04d}".format(i) + ".png"))
        i += 1
    
    # move all anamoly images to data_path and add 1_ prefix
    for anamoly_path in anamoly_paths:
        for filename in os.listdir(anamoly_path):
            os.rename(os.path.join(anamoly_path, filename), os.path.join(data_path, "1_" + "{:04d}".format(i) + ".png"))
            i += 1
    
    # remove all empty folders
    for anamoly_path in anamoly_paths:
        os.rmdir(anamoly_path)
    os.rmdir(normal_path)


def create_parser():
    defaults = dict(
        root=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "mvtec_anomaly_detection"),
        subset_name="",
    )
    parser = argparse.ArgumentParser()
    for k, v in defaults.items():
        parser.add_argument(f"--{k}", default=v, type=str)
    return parser

def main():
    args = create_parser().parse_args()
    if args.subset_name == "":
        for subset_name in MVTecAD.subset_names:
            preprocess_test_data(args.root, subset_name)
    else:
        preprocess_test_data(args.root, args.subset_name)

if __name__ == "__main__":
    main()