# ad-robust
Repository for the Robustness For Free: Adversarially Robust Anomaly Detection Through Diffusion Model paper

## installation

```
pip install numpy scipy opencv-python scikit-learn tqdm blobfile mpi4py
pip install -e .
```

## download and preprocess datasets
```
python ./scripts/mvtec_ad_download.py --root path/to/root [--subset_name NAME]
python ./scripts/preprocess_data.py --root path/to/root/mvtec_anomaly_dtection [--subset_name NAME]
```

## train
```
export DIFFUSION_BLOB_LOGDIR=path/to/saved/model
python ./scripts/image_train.py --data_dir path/to/train/good
```

## test
```
python ./scripts/image_test.py --data_dir path/to/test --model_path path/to/saved/model
```