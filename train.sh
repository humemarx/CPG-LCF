#！/bin/bash
GPUS=1
PORT=12099
CONFIG="config/nusc/config_e2e.py"
VERSION="nusc_cpg_lcf"

torchrun --nproc_per_node=$GPUS \
    --master_port=$PORT \
    train.py \
    --config $CONFIG \
    --version $VERSION \
    # --limit_train_batches 1 \
    # --limit_val_batches 1 \