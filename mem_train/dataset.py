import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

from topdown_map_utils import convert_maps_to_oh, generate_map_view_mask

import re
from constants import GIBSON_CATEGORIES

class MEM_build_Dataset(Dataset):
    local_map_size = (65, 65)
    map_resolution = 0.05
    pattern = r"location_\(\s*(\d+)\s*,\s*(\d+)\s*\)_direction_(\d+)"
    considered_semantic = GIBSON_CATEGORIES[1:]
    
    def __init__(self, root_dir):
        """
        Args:
            root_dir (str): Root directory containing subdirectories with h5 files.
        """
        self.root_dir = root_dir
        self.h5_files = []
        
        # Recursively search for 'local_data.h5' files.
        for dirpath, _, filenames in os.walk(self.root_dir):
            if "local_data.h5" in filenames:
                self.h5_files.append(os.path.join(dirpath, "local_data.h5"))
                
        if not self.h5_files:
            raise RuntimeError(f"No h5 files found in {root_dir}")

    def __len__(self):
        return len(self.h5_files)

    def get_sample_loc_dir(self, h5_path):
        match = re.search(self.pattern, h5_path)
        loc_x = int(match.group(1))
        loc_y = int(match.group(2))
        direction = int(match.group(3))
        return (loc_x, loc_y), direction

    def get_one_hot(self, sem_map):
        semmap_oh = convert_maps_to_oh(sem_map)
        return (semmap_oh.sum(axis=(1, 2)) > 0).astype(np.int8)

    def get_map_view_mask(self, local_map, init_dir):
        view_dir_mask = []
        for d_angle in [0, 90, 180, 270]:
            dir = (init_dir + d_angle)%360
            view_dir_mask.append(generate_map_view_mask(local_map, dir))
        return view_dir_mask

    def __getitem__(self, idx):
        h5_path = self.h5_files[idx]
        map_loc, map_dir = self.get_sample_loc_dir(h5_path)

        with h5py.File(h5_path, "r") as f:
            # Load the local map and RGB views.
            local_map = f["local_map"][:]    # e.g., shape (H, W) or (H, W, C)
            rgb_views = f["rgb_views"][:]      # e.g., shape (num_views, H, W, C)
            map_world_shift = f["map_world_shift"][:]
            if "blip2_embeds" in f:
                blip2_embeds = f["blip2_embeds"][:]
            else:
                blip2_embeds = None 

        local_map_oh = self.get_one_hot(local_map)
        views_mask = self.get_map_view_mask(local_map, map_dir)
        local_map = torch.from_numpy(local_map).unsqueeze(0).long()
        
        # Process rgb_views: assume shape is (num_views, H, W, C).
        views = []
        for i in range(rgb_views.shape[0]):
            # Convert each view from HWC to CHW and cast to float.
            view = torch.from_numpy(np.transpose(rgb_views[i], (2, 0, 1))).float()
            views.append(view)
        rgb_views = torch.stack(views, dim=0)
        
        if blip2_embeds:
            return {
                "local_map": local_map,    # Tensor, e.g., (1, H, W) for a 2D map.
                "rgb_views": rgb_views,    # Tensor, e.g., (num_views, C, H, W)
                "blip2_embeds":blip2_embeds,
            }
        else:
            return {
                "local_map": local_map,    # Tensor, e.g., (1, H, W) for a 2D map.
                "rgb_views": rgb_views,    # Tensor, e.g., (num_views, C, H, W)
            }

# Example usage:
if __name__ == "__main__":
    # Set your data directory.
    root_dir = "data/semantic_maps/gibson/image_map_pairs"
    
    dataset = MEM_build_Dataset(root_dir=root_dir)
    print("Total samples:", len(dataset))
    
    sample = dataset[0]
    print("Local map shape:", sample["local_map"].shape)   # e.g., (1, H, W)
    print("RGB views shape:", sample["rgb_views"].shape)     # e.g., (num_views, C, H, W)
