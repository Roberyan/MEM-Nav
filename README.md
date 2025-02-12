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