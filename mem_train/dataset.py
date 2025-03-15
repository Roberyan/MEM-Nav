import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from topdown_map_utils import convert_maps_to_oh, generate_map_view_mask
from tqdm import tqdm
import re
from PIL import Image
from constants import (
    GIBSON_CATEGORIES,
    OBJECT_CATEGORIES
)
    
class MEM_build_Dataset(Dataset):
    local_map_size = (65, 65)
    map_resolution = 0.05
    pattern = r"location_\(\s*(\d+)\s*,\s*(\d+)\s*\)_direction_(\d+)"
    considered_semantic = GIBSON_CATEGORIES[1:]
    view_angles = [0, 90, 180, 270]
    
    def __init__(self, root_dir, view_wise_oh=False, shuffle_views=False):
        """
        Args:
            root_dir (str): Root directory containing subdirectories with h5 files.
        """
        self.root_dir = root_dir
        self.h5_files = []
        self.shuffle_views = shuffle_views
        self.view_wise_oh = view_wise_oh
        # Recursively search for 'local_data.h5' files.
        for dirpath, _, filenames in os.walk(self.root_dir):
            if "local_data.h5" in filenames:
                self.h5_files.append(os.path.join(dirpath, "local_data.h5"))
                
        if not self.h5_files:
            raise RuntimeError(f"No h5 files found in {root_dir}")

    def __len__(self):
        return len(self.h5_files)

    # rule based loc and dir extraction from file path
    def get_sample_loc_dir(self, h5_path):
        match = re.search(self.pattern, h5_path)
        loc_x = int(match.group(1))
        loc_y = int(match.group(2))
        direction = int(match.group(3))
        return (loc_x, loc_y), direction

    # sem map to one hot map
    def get_one_hot(self, local_map, view_mask=None):
        sem_map = local_map if view_mask is None else np.where(view_mask, local_map, 0)
        semmap_oh = convert_maps_to_oh(sem_map)
        return (semmap_oh.sum(axis=(1, 2)) > 0).astype(np.int8)

    # topdown map local mask
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
            local_map = f["local_map"][:]    # e.g., shape (H, W) 
            rgb_views = f["rgb_views"][:]      # e.g., shape (num_views, H, W, C)
            # map_world_shift = f["map_world_shift"][:]
            if "blip2_embeds" in f:
                blip2_embeds = f["blip2_embeds"][:]
                blip2_embeds = torch.from_numpy(blip2_embeds).float()
            else:
                blip2_embeds = None 

        if self.view_wise_oh:
            # Generate one-hot vectors for each view using corresponding view masks.
            view_masks = self.get_map_view_mask(local_map, map_dir)
            oh_info = [torch.from_numpy(self.get_one_hot(local_map, mask)) for mask in view_masks]
            oh_info = torch.stack(oh_info, dim=0)
        else:
            oh_info = torch.from_numpy(self.get_one_hot(local_map)) # one-hot for whole local map

        # Convert local_map to torch tensor; assume it's a 2D map.
        local_map_tensor = torch.from_numpy(local_map).long()
        
        # Process rgb_views: convert from HWC to CHW and cast to float.
        rgb_views_tensor = torch.from_numpy(rgb_views).permute(0, 3, 1, 2).float()

        # shuffle views order while keeping the corresponding across [rgb_views, one-hot_views, embedding views]
        if self.shuffle_views:
            num_views = rgb_views_tensor.size(0)
            perm = torch.randperm(num_views)
            rgb_views_tensor = rgb_views_tensor[perm]
            oh_views = oh_views[perm]
            if blip2_embeds is not None:
                blip2_embeds = blip2_embeds[perm]
        
        sample_dict = {
            "local_map": local_map_tensor,      # Tensor, shape (1, H, W)
            "rgb_views": rgb_views_tensor,        # Tensor, shape (num_views, C, H, W)
            "onehot_info": oh_info,             # List of one-hot arrays, one per view.
            "h5_path": h5_path
        }
        if blip2_embeds is not None:
            sample_dict["blip2_embeds"] = blip2_embeds
        
        return sample_dict

# Example usage:
def test_dataset(root_dir):
    dataset = MEM_build_Dataset(root_dir=root_dir)
    print("Total samples:", len(dataset))
    cnt = 5
    for sample in tqdm(dataset):
        h5_file = sample["h5_path"]
        print("Local map shape:", sample["local_map"].shape)   # e.g., (1, H, W)
        print("RGB views shape:", sample["rgb_views"].shape)     # e.g., (num_views, C, H, W)
        # onehot_views is a list of one-hot vectors for each view.
        print("One-hot view vector shapes:", [oh.shape for oh in sample["onehot_views"]])
        if "blip2_embeds" in sample:
            print("blip2_embeds shape:", np.array(sample["blip2_embeds"]).shape)
        cnt -= 1
        if cnt<0:
            break

def pre_calculate_embeddings(root_path, nav_task="gibson", blip2_name="blip2_feature_extractor", recalculate_all=False):
    # Recursively search for 'local_data.h5' files.
    assert nav_task in root_path, f"Check if the path and nav task are corresponding!"
    from mem_vae_utils import load_blip2_model_lavis, generate_mem_prompt, load_instructblip_model_lavis

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if "instruct" in blip2_name:
        blip2_model, vis_processors, txt_processors = load_instructblip_model_lavis(blip2_name)
    else:
        blip2_model, vis_processors, txt_processors = load_blip2_model_lavis(blip2_name)
    # Move the model to the device.
    blip2_model.to(device)
    
    mem_prompt = generate_mem_prompt(OBJECT_CATEGORIES[nav_task])

    h5_files = []
    for dirpath, _, filenames in os.walk(root_path):
        if "local_data.h5" in filenames:
            h5_files.append(os.path.join(dirpath, "local_data.h5"))

    for h5_path in  tqdm(h5_files, desc=f"Precomputing blip2 embedding for mem_vae samples"):
        with h5py.File(h5_path, "a") as f:
            if not recalculate_all:
                if "blip2_embeds" in f:
                    continue
            rgb_views = f["rgb_views"][:]      # e.g., shape (num_views, H, W, C)
            # Process each view using the visual processor.
            processed_images = [vis_processors["eval"](Image.fromarray(img)).to(device) for img in rgb_views]
            # Stack into a batch: shape (num_views, C, H, W)
            images_batch = torch.stack(processed_images, dim=0).to(device)
            text_input = txt_processors["eval"](mem_prompt)
            # For BLIP2, text_input is expected as a list. We assume same prompt for all views.
            sample = {"image": images_batch, "text_input": [text_input]*4}
            # Extract features.
            if recalculate_all:
                if "blip2_embeds" in f:
                    del f["blip2_embeds"]
                    
            if "instruct" in blip2_name:
                blip2_embeds = blip2_model.get_qformer_features(sample)
                f.create_dataset("blip2_embeds", data=blip2_embeds.cpu().numpy(), compression="gzip")  
            else:
                blip2_embeds = blip2_model.extract_features(sample)
                f.create_dataset("blip2_embeds", data=blip2_embeds.multimodal_embeds.cpu().numpy(), compression="gzip")  

if __name__ == "__main__":
    # Set your data directory.
    root_dir = "data/semantic_maps/gibson/image_map_pairs"
    pre_calculate_embeddings(root_dir,  blip2_name="blip2_t5_instruct", recalculate_all=True)
    test_dataset(root_dir)
    