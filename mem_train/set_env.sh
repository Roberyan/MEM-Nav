export PONI_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:$PONI_ROOT


cd $PONI_ROOT
ACTIVE_DATASET="gibson" python mem_train/create_semantic_maps.py