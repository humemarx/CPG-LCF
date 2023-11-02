#！/bin/bash
GPUS=1
PORT=12099
CONFIG="config/nusc/config_e2e.py"
VERSION="nusc_cpg_lcf"
MODEL_PATH="experiments/07-25-14/config_e2e/nusc_cpg_lcf/checkpoint/bestmodel.ckpt"

torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train.py \
    --config $CONFIG \
    --version $VERSION \
    --is_validate \
    --pretrain_model $MODEL_PATH \