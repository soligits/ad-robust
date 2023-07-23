
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
        root="data",
        subset_name=""
    )

if __name__ == "__main__":
    main()