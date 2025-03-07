import os
import json
import glob
import random
import h5py
import numpy as np
import cv2
import habitat
import habitat_sim
import torch

import gc
import cv2
import bz2
import math
import json
import tqdm
import h5py
import glob
import torch
import random
import numpy as np
import os.path as osp
import _pickle as cPickle
import skimage.morphology as skmp

from PIL import Image, ImageFont, ImageDraw
from torch.utils.data import Dataset
from poni.geometry import (
    spatial_transform_map,
    crop_map,
    get_frontiers_np,
)
from poni.constants import (
    SPLIT_SCENES,
    OBJECT_CATEGORIES,
    INV_OBJECT_CATEGORY_MAP,
    NUM_OBJECT_CATEGORIES,
    # General constants
    CAT_OFFSET,
    # Coloring
    d3_40_colors_rgb,
    gibson_palette,
)
from poni.fmm_planner import FMMPlanner
from einops import asnumpy, repeat
from matplotlib import font_manager

# --- Paths and settings ---
SCENE_DIR = "data/scene_datasets/gibson_semantic"
SCENE_CONFIG = os.path.join(SCENE_DIR, "gibson_semantic.scene_dataset_config.json")
SEM_MAP_SAVE_ROOT = "data/semantic_maps/gibson/semantic_maps" 
SAVE_DIR = "data/semantic_maps/gibson/surrounding_images"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- From PONI --- 
MIN_OBJECTS_THRESH = 4
EPS = 1e-10

def is_int(s):
    try:
        int(s)
        return True
    except:
        return False

# -- visualization func from semantic map create
GIBSON_CATEGORIES = ["out-of-bounds"] + OBJECT_CATEGORIES["gibson"]
GIBSON_OBJECT_COLORS = [
    (0.9400000000000001, 0.7818, 0.66),
    (0.9400000000000001, 0.8868, 0.66),
    (0.8882000000000001, 0.9400000000000001, 0.66),
    (0.7832000000000001, 0.9400000000000001, 0.66),
    (0.6782000000000001, 0.9400000000000001, 0.66),
    (0.66, 0.9400000000000001, 0.7468000000000001),
    (0.66, 0.9400000000000001, 0.8518000000000001),
    (0.66, 0.9232, 0.9400000000000001),
    (0.66, 0.8182, 0.9400000000000001),
    (0.66, 0.7132, 0.9400000000000001),
    (0.7117999999999999, 0.66, 0.9400000000000001),
    (0.8168, 0.66, 0.9400000000000001),
    (0.9218, 0.66, 0.9400000000000001),
    (0.9400000000000001, 0.66, 0.8531999999999998),
    (0.9400000000000001, 0.66, 0.748199999999999),
]
GIBSON_COLOR_PALETTE = [
    1.0,
    1.0,
    1.0,  # Out-of-bounds
    0.9,
    0.9,
    0.9,  # Floor
    0.3,
    0.3,
    0.3,  # Wall
    *[oci for oc in GIBSON_OBJECT_COLORS for oci in oc],
]
GIBSON_LEGEND_PALETTE = [
    (1.0, 1.0, 1.0),  # Out-of-bounds
    (0.9, 0.9, 0.9),  # Floor
    (0.3, 0.3, 0.3),  # Wall
    *GIBSON_OBJECT_COLORS,
]
def get_palette_image():
    # Find a font file
    mpl_font = font_manager.FontProperties(family="sans-serif", weight="bold")
    file = font_manager.findfont(mpl_font)
    font = ImageFont.truetype(font=file, size=20)

    # Save color palette
    cat_size = 30
    buf_size = 10
    text_width = 150

    image = np.zeros(
        (cat_size * len(GIBSON_CATEGORIES), cat_size + buf_size + text_width, 3),
        dtype=np.uint8,
    )
    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    for i, (category, color) in enumerate(zip(GIBSON_CATEGORIES, GIBSON_LEGEND_PALETTE)):
        color = tuple([int(c * 255) for c in color])
        draw.rectangle(
            [(0, i * cat_size), (cat_size, (i + 1) * cat_size)],
            fill=color,
            outline=(0, 0, 0),
            width=2,
        )
        draw.text(
            [cat_size + buf_size, i * cat_size],
            category,
            font=font,
            fill=(255, 255, 255),
        )

    return np.array(image)

def visualize_sem_map(sem_map, selected_point=None, with_palette=True):
    c_map = sem_map.astype(np.int32)
    color_palette = [int(x * 255.0) for x in GIBSON_COLOR_PALETTE]
    semantic_img = Image.new("P", (c_map.shape[1], c_map.shape[0]))
    semantic_img.putpalette(color_palette)
    semantic_img.putdata((c_map.flatten() % 40).astype(np.uint8))
    semantic_img = semantic_img.convert("RGB")
    semantic_img = np.array(semantic_img)

    if selected_point is not None:
        cv2.circle(semantic_img, (int(selected_point[0]), int(selected_point[1])), 5, (255, 0, 0), thickness=-1)

    if with_palette:
        palette_img = get_palette_image()
        H = palette_img.shape[0]
        W = float(palette_img.shape[0]) * semantic_img.shape[1] / semantic_img.shape[0]
        W = int(W)
        semantic_img = cv2.resize(semantic_img, (W, H))
        semantic_img = np.concatenate([semantic_img, palette_img], axis=1)

    return semantic_img

# --- map data related funcs ---
FLOOR_ID = 1 # sem id, indicating navigatable area

def convert_maps_to_oh(semmap, dset="gibson"): # convert sem map to one hot, skip out-of-bound
    ncat = NUM_OBJECT_CATEGORIES[dset]
    semmap_oh = np.zeros((ncat, *semmap.shape), dtype=np.float32)
    for i in range(0, ncat):
        semmap_oh[i] = (semmap == i + CAT_OFFSET).astype(np.float32)
    return semmap_oh

def get_navigable_area_coordinates(sem_map, floor_label=FLOOR_ID):
    """
    Given a semantic map (2D array) and a floor label, return a list of pixel coordinates
    (x, y) where the map is navigable (i.e. floor).
    """
    y_coords, x_coords = np.where(sem_map == floor_label)
    # Return coordinates as (x, y) tuples
    return list(zip(x_coords, y_coords))

def pixel_to_world(pixel, resolution, world_shift, floor_y):
    """
    Convert a pixel coordinate (x, y) from the map to a 3D world coordinate.
    Here, x corresponds to the horizontal axis and y corresponds to the vertical axis in the map.
    """
    x, y = pixel
    world_x = x * resolution + world_shift[0]
    world_z = y * resolution + world_shift[2]
    return (world_x, floor_y, world_z)

def get_navigable_area_boundaries(sem_map, resolution, world_shift, floor_y, floor_label=FLOOR_ID):
    """
    Compute the boundaries (minimum and maximum corners) of the navigable area in world coordinates.
    """
    y_coords, x_coords = np.where(sem_map == floor_label)
    if len(x_coords) == 0 or len(y_coords) == 0:
        return None
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    world_min = pixel_to_world((min_x, min_y), resolution, world_shift, floor_y)
    world_max = pixel_to_world((max_x, max_y), resolution, world_shift, floor_y)
    return world_min, world_max


if __name__ == "__main__":
    grid_size = 0.05 # m
    object_boundary = 1.0 # m
    
    available_scenes = ['Allensville', 'Beechwood', 'Benevolence', 'Coffeen', 'Collierville', 'Cosmos', 'Darden', 'Forkland', 'Hanson', 'Hiteman', 'Klickitat', 'Lakeville', 'Leonardo', 'Lindenwood', 'Markleeville', 'Marstons', 'Merom', 'Mifflinburg', 'Newfields', 'Onaga', 'Pinesdale', 'Pomaria', 'Ranchester', 'Shelbyville', 'Stockman', 'Tolstoy', 'Wainscott', 'Wiconisco', 'Woodbine']

    maps_info = json.load(open(osp.join(SEM_MAP_SAVE_ROOT, 'semmap_GT_info.json')))

    dset = "gibson"

    maps = {}
    names = []
    maps_xyz_info = {}

    visibility_size = 3.0

    for scene_name in available_scenes:
        scene_sem_map_path =  os.path.join(SEM_MAP_SAVE_ROOT, f"{scene_name}.h5")
        scene_glb_path = os.path.join(SCENE_DIR, f'{scene_name}.glb')
        scene_navmesh_path =  os.path.join(SCENE_DIR, f'{scene_name}.navmesh')

        with h5py.File(scene_sem_map_path, 'r') as fp:
            floor_ids = sorted([key for key in fp.keys() if is_int(key)])
            for floor_id in floor_ids:
                name = f'{scene_name}_{floor_id}'
                map_world_shift = maps_info[scene_name]['map_world_shift']
                map_y = maps_info[scene_name][floor_id]['y_min']
                resolution = maps_info[scene_name]['resolution']
                map_semantic = np.array(fp[floor_id]['map_semantic'])

                # Convert to one-hot if needed (here we use the original sem_map)
                # semmap_oh = convert_maps_to_oh(map_semantic, dset=dset)

                # --- Compute available (navigable) area ---
                nav_coords_px = get_navigable_area_coordinates(map_semantic, floor_label=FLOOR_ID)
                if not nav_coords_px:
                    print(f"No navigable pixels found for {name}")
                    continue
                
                print(f"{len(nav_coords_px)} navigable positions")

                # Get boundaries (in world coordinates) of the navigable area:
                boundaries = get_navigable_area_boundaries(map_semantic, resolution, np.array(map_world_shift), map_y, floor_label=FLOOR_ID)
                if boundaries is not None:
                    world_min, world_max = boundaries
                    print(f"{name} navigable area boundaries (world coords):")
                    print(f"   Min: {world_min}")
                    print(f"   Max: {world_max}")
                else:
                    print(f"Could not determine boundaries for {name}")

                 # Optionally, convert a sampled pixel to world coordinate:
                
                sampled_pixel = random.choice(nav_coords_px)
                map_img_with_sampled_point = visualize_sem_map(map_semantic, sampled_pixel)
                cv2.imwrite("mem_train/test.png", map_img_with_sampled_point)
                
                sampled_world_coord = pixel_to_world(sampled_pixel, resolution, np.array(map_world_shift), map_y)
                print(f"{name} sampled navigable pixel {sampled_pixel} -> world position {sampled_world_coord}")

    print("==")