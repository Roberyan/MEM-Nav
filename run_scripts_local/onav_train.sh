#!/bin/bash

# Update PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/dependencies/habitat-sim

# Set log levels
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x
# For distributed training settings, if needed.
# For local runs without distributed training, these can be simplified:
export WORLD_SIZE=2
export NODE_RANK=-1
export LOCAL_RANK=-1

echo "Evaluating..."
echo "Hab-Sim: ${PYTHONPATH}"

# Choose the config file
# configfile=offline_bc/config/onav_rnn.yaml
# configfile=offline_bc/config/onav_transformer.yaml
configfile=offline_bc/config/onav_imap_single_transformer.yaml

# Run the training script directly
python offline_bc/train_local.py --exp-config $configfile
