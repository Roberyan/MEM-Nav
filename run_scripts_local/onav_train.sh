#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/dependencies/habitat-sim

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

set -x

# non-slurm servers
export WORLD_SIZE=1
export MASTER_ADDR='gpu003'
export MASTER_PORT=10000

echo "Evaluating..."
echo "Hab-Sim: ${PYTHONPATH}"

# configfile=offline_bc/config/onav_rnn.yaml
# configfile=offline_bc/config/onav_transformer.yaml
configfile=offline_bc/config/onav_imap_single_transformer.yaml

torchrun --nproc_per_node=1 offline_bc/train_local.py --exp-config $configfile

