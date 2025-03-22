
set -xe

# Change into project root
export PYTHONPATH=$PYTHONPATH:$(pwd)/dependencies/habitat-sim
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

configpath=habitat_baselines/config/objectnav/eval_LLM_models.yaml
val_dataset_path=data/datasets/objectnav/mp3d/v1
evalsplit=val

hf_llm=Qwen/Qwen2.5-VL-3B-Instruct
outdir=experiment_results/hf_llm_qwen
result_dir=${outdir}/eval_res
lora_checkpoint=""

python_bin=$(which python)

echo "Running evaluation with huggingface model ${hf_llm}"
${python_bin} -u habitat_baselines/run.py \
    --exp-config "${configpath}" \
    --run-type eval \
    TASK_CONFIG.DATASET.DATA_PATH "${val_dataset_path}/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.SENSORS "['OBJECTGOAL_SENSOR','COMPASS_SENSOR','GPS_SENSOR']" \
    EVAL_CKPT_PATH_DIR "${outdir}/ckpts" \
    CHECKPOINT_FOLDER "${outdir}/ckpts" \
    OUTPUT_LOG_DIR "${result_dir}/logs" \
    LOG_FILE "${result_dir}/valid.log" \
    TENSORBOARD_DIR "${result_dir}/tensorboad_dir" \
    VIDEO_DIR "${result_dir}/video_dir" \
    RESULTS_DIR "${result_dir}/results/sem_seg_pred/{split}/{type}" \
    EVAL_RESULTS_DIR "${result_dir}/results" \
    EVAL.SPLIT "${evalsplit}" \
    NUM_PROCESSES 1 \
    MODEL.model_class "${hf_llm}" \
    MODEL.if_hf_llm True \
    MODEL.lora_checkpoint "${lora_checkpoint}"