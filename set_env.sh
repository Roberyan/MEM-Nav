# set git account
git config user.name "Roberyan"
git config user.email "boyang.liu@u.nus.edu"

# data share without duplicate
ln -s "/media/marmot/One Touch/Boyang/data" data
# ln -s /media/data/Boyang/data data
rsync -avz  marmot@172.16.169.146:REMOTE_PATH LOCAL_PATH

# add habitat-sim to PYTHONPATH, for example add the following line to your .bashrc
export PYTHONPATH=$PYTHONPATH:$PWD/dependencies/habitat-sim
# avoid logging
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet

# change llm prompt in habitat_baselines/il/env_based/policy/hf_llm_models.py