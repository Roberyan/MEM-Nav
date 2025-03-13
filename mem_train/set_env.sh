git config --local user.name "Roberyan"
git config --local user.email "boyang.liu@u.nus.edu"

export PONI_ROOT=$PWD
export PYTHONPATH=$PYTHONPATH:$PONI_ROOT


cd $PONI_ROOT
ACTIVE_DATASET="gibson" python mem_train/create_semantic_maps.py

ln -s /home/marmot/Boyang/MEM-Nav/data /home/marmot/Boyang/Mcomp/MEM-Nav/data # data share without duplicate