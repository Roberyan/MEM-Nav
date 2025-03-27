import os
import h5py
import torch
from torch.utils.data import Dataset
import numpy as np
from topdown_map_utils import (
    convert_maps_to_oh, 
    generate_map_view_mask, 
    visualize_sem_map,
    rotate_local_map,
    smooth_semantic_map
)
from tqdm import tqdm
import re
from PIL import Image
from constants import (
    OBJECT_CATEGORIES
)
import shutil
import random
from constants import(
    FLOOR_ID,
    CAT_OFFSET,
    # Coloring
    d3_40_colors_rgb,
    gibson_palette,
)
import matplotlib.pyplot as plt

def custom_collate_fn(batch):
    batch_dict = {}
    batch_dict["local_map"] = torch.stack([sample["local_map"] for sample in batch], dim=0)
    batch_dict["rgb_views"] = torch.stack([sample["rgb_views"] for sample in batch], dim=0)
    batch_dict["depth_views"] = torch.stack([sample["depth_views"] for sample in batch], dim=0)
    batch_dict["onehot_info"] = torch.stack([sample["onehot_info"] for sample in batch], dim=0)

    # Only include "rgb_embeds" if every sample has it.
    if all("rgb_embeds" in sample and sample["rgb_embeds"] is not None for sample in batch):
        batch_dict["rgb_embeds"] = torch.stack([sample["rgb_embeds"] for sample in batch], dim=0)
    if all("depth_embeds" in sample and sample["depth_embeds"] is not None for sample in batch):
        batch_dict["depth_embeds"] = torch.stack([sample["depth_embeds"] for sample in batch], dim=0)
    
    return batch_dict

class MEM_build_Dataset(Dataset):
    local_map_size = (65, 65)
    map_resolution = 0.05
    pattern = r"location_\(\s*(\d+)\s*,\s*(\d+)\s*\)_direction_(\d+)"
    view_angles = [0, 90, 180, 270]
    max_depth=5.0
    
    def __init__(self, 
        root_dir="data/semantic_maps", 
        dataset="gibson",
        split= None,
        view_wise_oh=False, 
        shuffle_views=False,
        rotate_map=True,
        smooth_map=True
    ):
        self.dataset=dataset
        self.root_dir = os.path.join(root_dir, dataset, "image_map_pairs")
        print(f"Using datasets {dataset}.")
        
        if split is not None: # if prearrange train and test split
            self.root_dir = os.path.join(root_dir, split)
        
        self.h5_files = []
        self.shuffle_views = shuffle_views
        self.view_wise_oh = view_wise_oh
        self.rotate_map = rotate_map # make current map align with current view
        self.smooth_map = smooth_map
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
        semmap_oh = convert_maps_to_oh(sem_map, dset=self.dataset)
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
            depth_views = f["depth_views"][:]      # e.g., shape (num_views, H, W)
            # map_world_shift = f["map_world_shift"][:]
            if "rgb_embeds" in f:
                rgb_embeds = f["rgb_embeds"][:]
                rgb_embeds = torch.from_numpy(rgb_embeds).float()
            else:
                rgb_embeds = None 
            if "depth_embeds" in f:
                depth_embeds = f["depth_embeds"][:]
                depth_embeds = torch.from_numpy(depth_embeds).float()
            else:
                depth_embeds = None 

        # # for debugging
        # self.visualize_surrounding_views(start_angles=map_dir,rgb_views=rgb_views,depth_views=depth_views,save_path="/home/marmot/Boyang/MEM-Nav/tmp/dataset_views.png")
        
        if self.view_wise_oh:
            # Generate one-hot vectors for each view using corresponding view masks.
            view_masks = self.get_map_view_mask(local_map, map_dir)
            oh_info = [torch.from_numpy(self.get_one_hot(local_map, mask)) for mask in view_masks]
            oh_info = torch.stack(oh_info, dim=0)
        else:
            oh_info = torch.from_numpy(self.get_one_hot(local_map)) # one-hot for whole local map

        # Process rgb_views: convert from HWC to CHW and cast to float.
        rgb_views_tensor = torch.from_numpy(rgb_views).permute(0, 3, 1, 2).float()
        
        depth_views_tensor = torch.from_numpy(depth_views).float().unsqueeze(1)
        depth_views_tensor = depth_views_tensor.clamp(0, self.max_depth)

        # shuffle views order while keeping the corresponding order
        if self.shuffle_views:
            # pick a random offset [0 .. num_views‑1]
            shift = torch.randint(0, 4, (1,)).item()
            map_dir = (map_dir+90*shift)%360
            
            rgb_views_tensor = torch.roll(rgb_views_tensor, shifts=shift, dims=0)
            depth_views_tensor = torch.roll(depth_views_tensor, shifts=shift, dims=0)
            if self.view_wise_oh:
                oh_views = torch.roll(oh_views, shifts=shift, dims=0)
            if rgb_embeds is not None:
                rgb_embeds = torch.roll(rgb_embeds, shifts=shift, dims=0)
            if depth_embeds is not None:
                depth_embeds = torch.roll(depth_embeds, shifts=shift, dims=0)
        
        # Debugging, compare the map
        # self.visualize_local_map(local_map, map_dir, "/home/marmot/Boyang/MEM-Nav/tmp/dataset_map.png")
        # self.visualize_local_map(rotate_local_map(local_map, map_dir+90), -90, "/home/marmot/Boyang/MEM-Nav/tmp/dataset_map_rotated.png")
        # self.visualize_local_map(smooth_semantic_map(local_map, min_region_size= 4, max_hole_size= 4, majority_kernel= 3), map_dir, "/home/marmot/Boyang/MEM-Nav/tmp/dataset_map_smoothed.png")
        
        if self.smooth_map:
            local_map = smooth_semantic_map(
                local_map,
                min_region_size= 3,
                max_hole_size= 3,
                majority_kernel= 1
            )

        if self.rotate_map:
            local_map = rotate_local_map(local_map, map_dir+90) # current view as up

        local_map_tensor = torch.from_numpy(local_map).long()
        
        sample_dict = {
            "local_map": local_map_tensor,      # Tensor, shape (1, H, W)
            "rgb_views": rgb_views_tensor,        # Tensor, shape (num_views, C, H, W)
            "depth_views": depth_views,
            "onehot_info": oh_info,             # List of one-hot arrays, one per view.
            "h5_path": h5_path
        }
        if rgb_embeds is not None:
            sample_dict["rgb_embeds"] = rgb_embeds
        if depth_embeds is not None:
            sample_dict["depth_embeds"] = depth_embeds
        
        return sample_dict
    
    def visualize_local_map(self, 
        local_map, dir=None,
        save_path: str = None,
    ):
        vis=visualize_sem_map(local_map, dset=self.dataset, selected_point=np.array(self.local_map_size)//2, selected_angle=dir, with_info=False, with_palette=False)
        
        if save_path is not None:
            Image.fromarray(vis).save(save_path)
        else:
            Image.fromarray(vis).show()

    def visualize_surrounding_views(
        self,
        start_angles,
        rgb_views=None,
        depth_views=None,
        save_path: str = None,
        max_depth=5.0,
        figsize_per_col=(3, 3)
    ):
        N = 4
        rows = 1 + (rgb_views is not None) + (depth_views is not None)
        fig, axes = plt.subplots(
            rows, N, 
            figsize=(figsize_per_col[0]*N, figsize_per_col[1]*rows), 
            constrained_layout=True,
            squeeze=False,
            gridspec_kw={'height_ratios': [0.1] + [1]*(rows-1)}
        )

        # Top row = angle titles
        for i, angle in enumerate(self.view_angles):
            ax = axes[0, i]
            ax.axis("off")
            ax.text(0.5, 0.5, f"{(start_angles+angle)%360:.0f}°", ha="center", va="center", fontsize=20)

        row = 1
        if rgb_views is not None:
            for i in range(N):
                axes[row, i].imshow(rgb_views[i])
                axes[row, i].axis("off")
            row += 1

        if depth_views is not None:
            for i in range(N):
                d = np.clip(depth_views[i], 0, max_depth)
                im = axes[row, i].imshow(d, cmap="plasma", vmin=0, vmax=max_depth)
                axes[row, i].axis("off")
            # Single colorbar for the bottom depth row
            fig.colorbar(im, ax=axes[row, :], fraction=0.02, label="Depth (m)")

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        else:
            plt.show()

    def prepare_rgb_embedding_blip(self, vlm_name="blip2_t5_instruct", recalculate_all=True):
        assert vlm_name is not None, "You must choose a vlm model to run do the calculation"
        from mem_vae_utils import (
            load_blip2_model_lavis, 
            generate_mem_prompt, 
            load_instructblip_model_lavis
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
        if "instruct" in vlm_name:
            blip2_model, vis_processors, txt_processors = load_instructblip_model_lavis(vlm_name)
        else:
            blip2_model, vis_processors, txt_processors = load_blip2_model_lavis(vlm_name)
        # Move the model to the device.
        blip2_model.to(device)
    
        mem_prompt = generate_mem_prompt(OBJECT_CATEGORIES[self.dataset])
        text_input = txt_processors["eval"](mem_prompt)
        
        for h5_path in  tqdm(self.h5_files, desc=f"Precomputing {vlm_name} embedding for mem_vae samples"):
            with h5py.File(h5_path, "a") as f:
                if not recalculate_all:
                    if "rgb_embeds" in f:
                        continue
                else:
                    if "rgb_embeds" in f:
                        del f["rgb_embeds"]

                rgb_views = f["rgb_views"][:]      # e.g., shape (num_views, H, W, C)
                # Process each view using the visual processor.
                processed_images = [vis_processors["eval"](Image.fromarray(img)).to(device) for img in rgb_views]
                # Stack into a batch: shape (num_views, C, H, W)
                images_batch = torch.stack(processed_images, dim=0).to(device)
                # For BLIP2, text_input is expected as a list. We assume same prompt for all views.
                sample = {"image": images_batch, "text_input": [text_input]*4}
                # Extract features.
                
                if "instruct" in vlm_name:
                    embeds = blip2_model.get_qformer_features(sample)
                else:
                    embeds = blip2_model.extract_features(sample).multimodal_embeds
                
                f.create_dataset("rgb_embeds", data=embeds.cpu().numpy(), compression="gzip")

    def prepare_rgb_embedding_clip(self, recalculate_all=True):
        from mem_vae_utils import CLIPEncoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rgb_clip_encoder = CLIPEncoder(device)
        
        for h5_path in  tqdm(self.h5_files, desc=f"Precomputing clip embedding for mem_vae samples"):
            with h5py.File(h5_path, "a") as f:
                if not recalculate_all:
                    if "rgb_embeds" in f:
                        continue
                else:
                    if "rgb_embeds" in f:
                        del f["rgb_embeds"]

                rgb_views = f["rgb_views"][:]      # e.g., shape (num_views, H, W, C)
                flat_imgs = [Image.fromarray(view) for view in rgb_views]
                clip_feats = rgb_clip_encoder.extract_fts(flat_imgs)
                f.create_dataset("rgb_embeds", data=clip_feats, compression="gzip")

    def prepare_depth_embedding(self, depth_name="gibson-2plus-resnet50.pth", recalculate_all=True):
        from mem_vae_utils import DepthEncoder
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        depth_encoder = DepthEncoder(device)
        
        for h5_path in  tqdm(self.h5_files, desc=f"Precomputing {depth_name} embedding for mem_vae samples"):
            with h5py.File(h5_path, "a") as f:
                if not recalculate_all:
                    if "depth_embeds" in f:
                        continue
                else:
                    if "depth_embeds" in f:
                        del f["depth_embeds"]

                depth_views = f["depth_views"][:]
                depth_batch = torch.from_numpy(depth_views).unsqueeze(-1).to(device)
                depth_embeds = depth_encoder.extract_fts(depth_batch)  # shape (4, 2048)
                f.create_dataset("depth_embeds", data=depth_embeds, compression="gzip")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Precompute RGB & Depth embeddings for MEM dataset")
    parser.add_argument("--root_dir", type=str, default="data/semantic_maps", help="Path to semantic maps root")
    parser.add_argument("--dataset", type=str, default="mp3d", choices=["mp3d", "gibson"], help="Which dataset to build")
    parser.add_argument("--precompute_depth", action="store_true", help="Compute depth embeddings")
    parser.add_argument("--precompute_rgb_blip", action="store_true", help="Compute RGB embeddings")
    parser.add_argument("--precompute_rgb_clip", action="store_true", help="Compute RGB embeddings")
    args = parser.parse_args()
    
    # Set your data directory.
    dataset = MEM_build_Dataset(
        root_dir=args.root_dir,
        dataset=args.dataset
    )
    print("Total samples:", len(dataset))
    if args.precompute_depth:
        print("→ Preparing depth embeddings…")
        dataset.prepare_depth_embedding()

    if args.precompute_rgb_blip:
        print("→ Preparing blip RGB embeddings…")
        dataset.prepare_rgb_embedding_blip()
    
    if args.precompute_rgb_clip:
        print("→ Preparing clip RGB embeddings…")
        dataset.prepare_rgb_embedding_clip()