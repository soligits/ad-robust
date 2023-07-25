# ad-robust
Repository for the Robustness For Free: Adversarially Robust Anomaly Detection Through Diffusion Model paper

## installation

```
pip install numpy scipy opencv-python scikit-learn tqdm blobfile mpi4py
pip install -e .
```

## download and preprocess datasets
```
export PREPARE_DATA="[--subset_name SUBSET_NAME] [--root ROOT]"
python scripts/prepare_data.py PREPARE_DATA
```

## train
```
export DIFFUSION_BLOB_LOGDIR=path/to/saved/model
python ./scripts/image_train.py --data_dir path/to/train/good
```

## test
```
```