import os
import argparse

from ad_utils.mvtec_ad import MVTecAD

def main():
    args = create_argparser().parse_args()

    # download data
    if args.subset_name == "":
        for subset_name in MVTecAD.subset_names:
            MVTecAD(
                args.root,
                subset_name=subset_name,
                download=True,
            )
        os.remove(os.path.join(args.root, subset_name, ".tar.xz"))
    elif args.subset_name in MVTecAD.subset_names:
        MVTecAD(
            args.root,
            subset_name=args.subset_name,
            download=True,
        )
    else:
        raise ValueError(f"Invalid subset_name: {args.subset_name}")

def create_argparser():
    defaults = dict(
        root=os.path.join(os.path.dirname(os.path.abspath(__file__)),"..", "data"),
        subset_name=""
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=defaults["root"])
    parser.add_argument("--subset_name", type=str, default=defaults["subset_name"])
    return parser

    

if __name__ == "__main__":
    main()