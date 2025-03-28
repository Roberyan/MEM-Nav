#!/bin/bash
export PYTHONPATH=$PYTHONPATH:$(pwd)/dependencies/habitat-sim

python offline_bc/preprocess/extract_demo_observations.py   --cfg_file habitat_baselines/config/objectnav/il_ddp_objectnav.yaml   --scene_id all --save_topdown_map --save_semantic_fts --encode_depth --encode_rgb_clip  --encode_depth_views --encode_rgb_views_clip --save_rgb  --outdir data/datasets/objectnav/mp3d_70k_demos_prefts