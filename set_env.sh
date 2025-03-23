# set git account
git config user.name "Roberyan"
git config user.email "boyang.liu@u.nus.edu"

# data share without duplicate
ln -s "/media/marmot/One Touch/Boyang/data" data
# delete views png to save memory
find data/semantic_maps/gibson/image_map_pairs -type f -name 'Panorama*.png' -delete 

# set root path
export PONI_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:$PONI_ROOT
cd $PONI_ROOT

# build semantic map
ACTIVE_DATASET="gibson" python mem_train/create_semantic_maps.py
# sample from sim to extract local views map pairs
ACTIVE_DATASET="mp3d" python mem_train/create_image_map.py
# precompute blip2 embeddings, currently just in dataset.py, need to arrange later
python mem_train/dataset.py 

# extract FMM distances for all objects in each map
ACTIVE_DATASET="gibson" python nav_train/precompute_fmm_dists.py
# Extract training and validation data for PONI
ACTIVE_DATASET="gibson" python nav_train/create_poni_dataset.py --split "train"
ACTIVE_DATASET="gibson" python nav_train/create_poni_dataset.py --split "val"