import json
from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from nav_vlm.constant import SEM_MAP_SAVE_ROOT
import numpy as np
import h5py
import random

class NavDemoDataset(Dataset):
    def __init__(self, 
            root_dir,
            scene_id=None,
            demo_merge_ratio=0.5,
            episode_merge_ratio=0.5,
            p_merge_all_episode=0.2,
            max_acts_per_episode=500
        ):
        self.root_dir = root_dir
        available_scenes = self.get_available_scenes()
        self.topdown_map_root = SEM_MAP_SAVE_ROOT
        self.demo_merge_ratio = demo_merge_ratio
        self.episode_merge_ratio = episode_merge_ratio
        self.p_merge_all_episode = p_merge_all_episode
        self.max_acts_per_episode = max_acts_per_episode
        
        if scene_id:
            if isinstance(scene_id, str):
                if scene_id in available_scenes:
                    self.used_scenes = [scene_id]
            elif isinstance(scene_id, list):
                self.used_scenes = []
                for scene in scene_id:
                    if scene in available_scenes:
                        self.used_scenes.append(scene_id)
        else:
            self.used_scenes = available_scenes
        
        self.demos = []
        self.load_from_used_scene()
    
    def __len__(self):
        return len(self.demos)
    
    def get_available_scenes(self):
        return os.listdir(self.root_dir)

    def get_scene_ann_file(self, scene):
        return os.path.join(self.root_dir, scene, "annotations.jsonl")
    
    def load_from_used_scene(self):
        self.demos.clear()
        for scene in self.used_scenes:
            ann_file = self.get_scene_ann_file(scene)
            with open(ann_file, "r") as f:
                for line in f:
                    self.demos.append(json.loads(line))
    
    def set_scene(self, scene_id):
        available_scenes = self.get_available_scenes()
        if isinstance(scene_id, str):
            if scene_id in available_scenes:
                self.used_scenes = [scene_id]
            self.load_from_used_scene()
        elif isinstance(scene_id, list):
            self.used_scenes = []
            for scene in scene_id:
                if scene in available_scenes:
                    self.used_scenes.append(scene_id)
            self.load_from_used_scene()
        else:
            print("Invalid scene id, keep original used ids")            

    def load_topdown_sem_map(self, scene_name, floor_id):
        maps_info = json.load(open(os.path.join(SEM_MAP_SAVE_ROOT, 'semmap_GT_info.json')))
        scene_sem_map_path =  os.path.join(SEM_MAP_SAVE_ROOT, f"{scene_name}.h5")
        map_world_shift = maps_info[scene_name]['map_world_shift']
        map_resolution = maps_info[scene_name]['resolution']
        with h5py.File(scene_sem_map_path, "r") as fp:
            # map_y = maps_info[scene_name][floor_id]['y_min']
            map_semantic = np.array(fp[floor_id]['map_semantic'])
        
        return map_world_shift, map_resolution, map_semantic
    
    def combine_turn_forward(self, demo_episode, allow_full_merge=True):
        stop_id = len(demo_episode['demonstration']) - 1
        merge_id = []
        for i, act in enumerate(demo_episode['demonstration']):
            if "TURN" in act[0]:
                merge_id.append(i)
        
        if not allow_full_merge:
            merge_id = random.sample(merge_id, int(self.episode_merge_ratio * len(merge_id)))
        else:
            if random.random() > self.p_merge_all_episode:
                merge_id = random.sample(merge_id, int(self.episode_merge_ratio * len(merge_id)))
        
        for step in merge_id:
            demo_episode['demonstration'][step].extend(demo_episode['demonstration'][step+1])
            demo_episode['reward'][step] += demo_episode['reward'][step+1]
            demo_episode['info'][step] = demo_episode['info'][step+1]
        
        merged_steps = [i+1 for i in merge_id]
        if stop_id in merged_steps:
            merged_steps.remove(stop_id)

        for k, v in demo_episode.items():
            if k in ['episode_id', 'floor_id', 'objectgoal', 'object_category']:
                continue
            demo_episode[k] = [v[idx] for idx in range(len(v)) if idx not in merged_steps]
        
        return demo_episode
    
    @ staticmethod
    def remove_long_demos(demos, max_act=-1):
        max_act = max_act if max_act > 0 else self.max_acts_per_episode
        return [
            demo for demo in demos 
            if len(demo['demonstration']) <= max_act
        ]

    def get_demos(self):
        return self.demos
    
    def __getitem__(self, idx):
        demo = self.demos[idx] # one whole episodes
        if random.random() < self.demo_merge_ratio:
            demo = self.combine_turn_forward(demo)
        return demo

if __name__ == "__main__":
    ROOT_DIR = "/home/marmot/Boyang/MEM-Nav/data/datasets/objectnav/mp3d_70k_demos_for_vlm"
    SCENE = "17DRP5sb8fy"
    
    dataset = NavDemoDataset(
        root_dir=ROOT_DIR, 
        scene_id=SCENE
    )
    
    print("Number of demo episodes:", len(dataset))