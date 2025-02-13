# MEM-Nav

## Installation
```bash
conda_env_name=mem-nav
conda create -n $conda_env_name python=3.9 cmake=3.14.0 -y
conda activate $conda_env_name

# habitat-sim
conda install habitat-sim=0.2.4 withbullet -c conda-forge -c aihabitat

# torch
conda install pytorch==1.12 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
pip install torch==1.12+cu102 torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html

# other package
pip install salesforce-lavis==1.0.2 transformers==4.26.0 numpy==1.26.4 imageio-ffmpeg pillow==10.4.0
pip install tqdm wandb tensorboard

# habitat-lab
git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab/
git checkout tags/v0.2.4
pip install -e habitat-lab
pip install -e habitat-baselines
```

## Dataset Download

Recommend to create a folder and download data in the folder
```bash
mkdir -p data
cd data
```

General Usage of habitat to download data
```bash 
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path ./

python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path ./
```

### Download HM3D Dataset
[official link for scene data](https://github.com/facebookresearch/habitat-sim/blob/main/DATASETS.md#habitat-matterport-3d-research-dataset-hm3d)
[official link for task data](https://github.com/facebookresearch/habitat-lab/blob/main/DATASETS.md#task-datasets)

``` bash
# scene data downloading
MATTERPORT_TOKEN_ID=ead2abaa44071dd5
MATTERPORT_TOKEN_SECRET=dfe9bcd56c2eecd2092de895f95ba4cc
DATA_DIR="./"

python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_minival_v0.2 \
  --data-path $DATA_DIR 

python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_train_v0.2 \
  --data-path $DATA_DIR 

python -m habitat_sim.utils.datasets_download \
  --username $MATTERPORT_TOKEN_ID --password $MATTERPORT_TOKEN_SECRET \
  --uids hm3d_val_v0.2 \
  --data-path $DATA_DIR 

# objnav data downloading
wget https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/hm3d/v1/objectnav_hm3d_v1.zip
unzip objectnav_hm3d_v1.zip
mkdir -p ./datasets/objectnav/hm3d
mv objectnav_hm3d_v1 ./datasets/objectnav/hm3d/v1
rm objectnav_hm3d_v1.zip
```

### Download Gibson Dataset
[official link for scene data](https://github.com/facebookresearch/habitat-lab?tab=readme-ov-file#scenes-datasets)
[task data from SGM](https://github.com/sx-zhang/SGM/tree/main)

```bash
# objnav data downloading
wget -O gibson_objectnav_episodes.tar.gz https://utexas.box.com/shared/static/tss7udt3ralioalb6eskj3z3spuvwz7v.gz
tar -xvzf gibson_objectnav_episodes.tar.gz && rm gibson_objectnav_episodes.tar.gz
mv gibson datasets/objectnav
```