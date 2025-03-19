# set git account
git config user.name "Roberyan"
git config user.email "boyang.liu@u.nus.edu"

# data share without duplicate
ln -s "/media/marmot/One Touch/Boyang/data" data

# add habitat-sim to PYTHONPATH, for example add the following line to your .bashrc
export PYTHONPATH=$PYTHONPATH:$PWD/dependencies/habitat-sim
# avoid logging
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet