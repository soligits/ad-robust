# ad-robust
Repository for the Robustness For Free: Adversarially Robust Anomaly Detection Through Diffusion Model paper

## installation

```
pip install numpy scipy opencv-python scikit-learn tqdm blobfile mpi4py
pip install torch torchvision
pip install -e .
```

## download and preprocess datasets
```
PREPARE_DATA="[--subset_name SUBSET_NAME] [--root ROOT]"
python scripts/prepare_data.py PREPARE_DATA
```

## train
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3 --num_heads 1 --attention_resolutions 8"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 2 --lr_anneal_steps 30000"
DATA_FLAGS="--data_dir /path/to/data/train"
export DIFFUSION_BLOB_LOGDIR=path/for/saving/model/
python ./scripts/image_train.py MODEL_FLAGS DIFFUSION_FLAGS TRAIN_FLAGS DATA_FLAGS
```

## test
```
```