
## env installation
```
conda_env_name=mem-nav
conda create -n $conda_env_name python=3.9 cmake=3.14.0 -y
conda activate $conda_env_name

conda install -c conda-forge imageio-ffmpeg

git clone https://github.com/facebookresearch/habitat-sim.git
cd habitat-sim; git checkout tags/challenge-2022;
pip install -r requirements.txt;
python setup.py install --headless

git clone https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab; git checkout tags/challenge-2022; 
pip install -e .
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

```bash
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

### Download MP3D Dataset

```
Thank you for your interest in Matterport3D. Please use the following script to download the Matterport3D data: http://kaldir.vc.in.tum.de/matterport/download_mp.py. 

Some useful info:
Scan data is named by a house hash id. The list of house hash ids is at http://kaldir.vc.in.tum.de/matterport/v1/scans.txt 
Script usage:
- To download the entire Matterport3D release (1.3TB): download-mp.py -o [directory in which to download] 
- To download a specific scan (e.g., 17DRP5sb8fy): download-mp.py -o [directory in which to download] --id 17DRP5sb8fy
- To download a specific file type (e.g., *.sens, valid file suffixes listed here): download-mp.py -o [directory in which to download] --type .sens
- *.sens files can be read using the sens-File reader (it's a bit easier to handle than a larger number of separate image files)

License: Matterport3D data is released under the Terms of Use. By downloading the data, you agree to these terms!
```
