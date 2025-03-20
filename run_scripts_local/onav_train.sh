#!/bin/bash

# Update PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/dependencies/habitat-sim

# Set log levels
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x
# For distributed training settings, if needed.
# For local runs without distributed training, these can be simplified:
export WORLD_SIZE=1
export MASTER_ADDR='gpu003'
export MASTER_PORT=10000
export NODE_RANK=3
export LOCAL_RANK=3

echo "Evaluating..."
echo "Hab-Sim: ${PYTHONPATH}"

# Choose the config file
# configfile=offline_bc/config/onav_rnn.yaml
# configfile=offline_bc/config/onav_transformer.yaml
configfile=offline_bc/config/onav_imap_single_transformer.yaml

# Run the training script directly
export CUDA_VISIBLE_DEVICES=1,2,3
python offline_bc/train_local.py --exp-config $configfile
