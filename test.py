import sys,pdb,os,time
import gzip
import json

import glob
import numpy as np

import habitat
from habitat.sims import make_sim
from habitat_sim import Simulator
import habitat_sim

from constants import mp3d_category_id, category_to_id, hm3d_category

fileName = '../data/matterport_category_mappings.tsv'

text = ''
lines = []
items = []
hm3d_semantic_mapping={}

with open(fileName, 'r') as f:
    text = f.read()
lines = text.split('\n')

for l in lines:
    items.append(l.split('    '))

for i in items:
    if len(i) > 3:
        hm3d_semantic_mapping[i[2]] = i[-1]

def task_data_check_demo():
    data_pth='../data/datasets/objectnav/hm3d/v1/val_mini/content/TEEsavR23oF.json.gz'
    with gzip.open(data_pth, 'r') as f:
        data = json.loads(f.read().decode('utf-8'))
    for k in data.keys():
        print(k,len(data[k]))
    print("========================")
    print(data['episodes'][1])
    print("========================")
    print(data['episodes'][2])
    print("========================")
    print(data['episodes'][3])


def extract_objects_from_room_demo(SPLIT="val"):
    objects = []
    labels = []
    all_objs = []

    data_generation_params = [(1, 1), (2, 2), (3, 3), (1, 2), (2, 3), (3, 4)]
    max_n = np.max([i[1] for i in data_generation_params])
    max_num_obj = max_n

    scenes = glob.glob("../data/scene_datasets/hm3d/"+SPLIT+"/*/*.basis.glb")
    dataset_path = glob.glob("../data/datasets/objectnav/hm3d/v1/"+SPLIT+"/content/*.json.gz")
    split_dataset_path = []
    for datasets in dataset_path:
        split_dataset_path.append(os.path.split(datasets)[1].split('.')[0])
    print(scenes)
    print(len(scenes))




if __name__ == "__main__":
    t=time.time()

    # task_data_check_demo()
    extract_objects_from_room_demo()
    
    end_t=time.time()
    print(f"duration: {end_t-t} s")