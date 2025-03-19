#!/bin/bash
# Usage: ./run_eval_local.sh <output_dir> <checkpoint_step>

set -xe

# Change into project root

export PYTHONPATH=$PYTHONPATH:$(pwd)/dependencies/habitat-sim
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

outdir=$1
ckpt_step=$2
result_dir=${outdir}/step_${ckpt_step}_heuristic_nocld

configpath=habitat_baselines/config/objectnav/eval_imap_single_transformer.yaml
val_dataset_path=data/datasets/objectnav/mp3d/v1
checkpoint=${outdir}/ckpts/model_step_${ckpt_step}.pt
evalsplit=val

# Use whichever Python is in your onav env
python_bin=$(which python)

echo "Running evaluation with checkpoint step ${ckpt_step} → ${result_dir}"
${python_bin} -u habitat_baselines/run.py \
  --exp-config "${configpath}" \
  --run-type eval \
  TASK_CONFIG.DATASET.DATA_PATH "${val_dataset_path}/{split}/{split}.json.gz" \
  TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR','COMPASS_SENSOR','GPS_SENSOR']" \
  EVAL_CKPT_PATH_DIR "${checkpoint}" \
  CHECKPOINT_FOLDER "${outdir}/ckpts" \
  OUTPUT_LOG_DIR "${outdir}/logs" \
  LOG_FILE "${result_dir}/valid.log" \
  TENSORBOARD_DIR "${result_dir}/tb" \
  VIDEO_DIR "${result_dir}/video_dir" \
  RESULTS_DIR "${result_dir}/results/sem_seg_pred/{split}/{type}" \
  EVAL_RESULTS_DIR "${result_dir}/results" \
  EVAL.USE_CKPT_CONFIG False \
  EVAL.SPLIT "${evalsplit}" \
  NUM_PROCESSES 1 \
  EVAL_CKPT_FROM_OFFLINEBC True \
  MODEL.encoder_type concat \
  MODEL.encoder_add_objgoal True \
  MODEL.enc_collide_steps False \
  MODEL.MAP_ENCODER.imap_size 3 \
  MODEL.MAP_ENCODER.token_embed_type single \
  MODEL.MAP_ENCODER.encode_position True \
  MODEL.STATE_ENCODER.add_pos_attn True
