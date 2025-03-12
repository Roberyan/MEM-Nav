import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
import random
from topdown_map_utils import convert_maps_to_oh, generate_map_view_mask

import re
from constants import (
    GIBSON_CATEGORIES,
    INV_OBJECT_CATEGORY_MAP
)

def shuffle_views(view_info_groups):
    indices = list(range(len(view_info_groups[0])))
    random.shuffle(indices)

    for idx in range(len(view_info_groups)):
        view_info_groups[idx] = [view_info_groups[idx][i] for i in indices]

    return view_info_groups

class MEM_build_Dataset(Dataset):
    local_map_size = (65, 65)
    map_resolution = 0.05
    pattern = r"location_\(\s*(\d+)\s*,\s*(\d+)\s*\)_direction_(\d+)"
    considered_semantic = GIBSON_CATEGORIES[1:]
    view_angles = [0, 90, 180, 270]
    
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

    def get_one_hot(self, local_map, view_mask=None):
        sem_map = local_map if view_mask is None else np.where(view_mask, local_map, 0)
        semmap_oh = convert_maps_to_oh(sem_map)
        return (semmap_oh.sum(axis=(1, 2)) > 0).astype(np.int8)

    def get_map_view_mask(self, local_map, init_dir):
        view_dir_mask = []
        for d_angle in self.view_angles:
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
            # map_world_shift = f["map_world_shift"][:]
            if "blip2_embeds" in f:
                blip2_embeds = f["blip2_embeds"][:]
                blip2_embeds = torch.from_numpy(blip2_embeds)
            else:
                blip2_embeds = None 

        # Generate one-hot vectors for each view using corresponding view masks.
        view_masks = self.get_map_view_mask(local_map, map_dir)
        oh_views = [self.get_one_hot(local_map, mask) for mask in view_masks]

        # local_map_oh = self.get_one_hot(local_map)

        # Convert local_map to torch tensor; assume it's a 2D map.
        local_map_tensor = torch.from_numpy(local_map).unsqueeze(0).long()
        
        # Process rgb_views: convert from HWC to CHW and cast to float.
        rgb_views_tensor = torch.from_numpy(rgb_views).permute(0, 3, 1, 2).float()
        
        sample_dict = {
            "local_map": local_map_tensor,      # Tensor, shape (1, H, W)
            "rgb_views": rgb_views_tensor,        # Tensor, shape (num_views, C, H, W)
            "onehot_views": oh_views,             # List of one-hot arrays, one per view.
            "h5_path": h5_path
        }
        if blip2_embeds is not None:
            sample_dict["blip2_embeds"] = blip2_embeds
        
        return sample_dict

# Example usage:
if __name__ == "__main__":
    # Set your data directory.
    root_dir = "data/semantic_maps/gibson/image_map_pairs"
    
    dataset = MEM_build_Dataset(root_dir=root_dir)
    print("Total samples:", len(dataset))
    
    sample = dataset[0]
    print("Local map shape:", sample["local_map"].shape)   # e.g., (1, H, W)
    print("RGB views shape:", sample["rgb_views"].shape)     # e.g., (num_views, C, H, W)
    # onehot_views is a list of one-hot vectors for each view.
    print("One-hot view vector shapes:", [oh.shape for oh in sample["onehot_views"]])
    if "blip2_embeds" in sample:
        print("blip2_embeds shape:", np.array(sample["blip2_embeds"]).shape)