#！/bin/bash
GPUS=1
PORT=12099
CONFIG="config/nusc/config_kd_e2e.py"
VERSION="nusc_cpg_lcf_kd"
MODEL_PATH="experiments/07-25-14/config_e2e/nusc_cpg_lcf/checkpoint/bestmodel.ckpt"

torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train.py \
    --config $CONFIG \
    --version $VERSION \
    --pretrain_model $MODEL_PATH \
    # --limit_train_batches 1 \
    # --limit_val_batches 1 \