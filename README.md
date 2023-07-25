# ad-robust
Repository for the Robustness For Free: Adversarially Robust Anomaly Detection Through Diffusion Model paper

## installation

```
pip install -r requirements.txt
pip install -e .
```

## download and preprocess datasets
```
PREPARE_DATA="[--subset_name SUBSET_NAME] [--root ROOT]"
python scripts/prepare_data.py PREPARE_DATA
```

## train
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_heads 1 --attention_resolutions 8"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --lr_anneal_steps 30000"
DATA_FLAGS="--data_dir /path/to/data/train --batch_size 2"
export DIFFUSION_BLOB_LOGDIR=path/for/saving/model/
python scripts/image_train.py MODEL_FLAGS DIFFUSION_FLAGS TRAIN_FLAGS DATA_FLAGS
```

## test
```
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_heads 1 --attention_resolutions 8 --model_path /path/to/model/chosen_model.pt"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
DATA_FLAGS="--test_path /path/to/data/test --train_path /path/to/data/train --batch_size 1"
AD_FLAGS="--k_steps K --m_shot M --anomaly_threshold H --mean_filter_size S"
ATTACK_FLAGS="--attack_type ATK_TYPE --attack_strength EPSILON --attack_n N --attack_alpha ALPHA"

python scripts/image_test.py MODEL_FLAGS DIFFUSION_FLAGS DATA_FLAGS AD_FLAGS ATTACK_FLAGS
```