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
import habitat_sim.utils.common as common_utils

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
from envs import init_sim
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

def visualize_sem_map(sem_map, selected_point=None, selected_angle=None, with_palette=True):
    """
    Visualize the semantic map using the Gibson color palette and overlay:
      - A coordinate legend in the top-left corner (drawn in red) indicating +X (arrow right) and +Z (arrow downward),
      - A title at the top-center showing "Location: (x, y), Direction: xx°" in black,
      - And, if provided, an oriented red triangle marker at the selected position.
    
    The overlays are drawn directly on the semantic map so that the map itself (before appending the palette)
    retains its original size.
    
    Args:
        sem_map: 2D numpy array with semantic class labels.
        selected_point: Tuple (x, y) in the original sem_map coordinate space.
        selected_angle: Heading angle in degrees (0° means pointing east/right).
        with_palette: If True, appends the palette image as extra information.
    
    Returns:
        An RGB image (numpy array) with overlays drawn on top of the original map.
    """
    # Convert semantic map to an RGB image using the Gibson palette.
    c_map = sem_map.astype(np.int32)
    color_palette = [int(x * 255.0) for x in GIBSON_COLOR_PALETTE]
    semantic_pil = Image.new("P", (c_map.shape[1], c_map.shape[0]))
    semantic_pil.putpalette(color_palette)
    semantic_pil.putdata((c_map.flatten() % 40).astype(np.uint8))
    semantic_pil = semantic_pil.convert("RGB")
    
    # Create a drawing context on the PIL image.
    draw = ImageDraw.Draw(semantic_pil)
    
    # Load a bold sans-serif font at size 14.
    font_path = font_manager.findfont(font_manager.FontProperties(family="sans-serif", weight="bold"))
    custom_font = ImageFont.truetype(font_path, 14)
    
    # --- Draw title text (top-center) ---
    if selected_point is not None and selected_angle is not None:
        title_text = f"Location: ({selected_point[0]}, {selected_point[1]}), Direction: {selected_angle}°"
        w, h = semantic_pil.size
        # Use textbbox to get the bounding box of the text.
        bbox = draw.textbbox((0, 0), title_text, font=custom_font)
        text_width = bbox[2] - bbox[0]
        title_x = (w - text_width) // 2  # center horizontally
        title_y = 5  # top margin
        draw.text((title_x, title_y), title_text, font=custom_font, fill=(0, 0, 0))
    
    # --- Draw coordinate legend (top-left), shifted down to avoid title overlap ---
    legend_origin = (20, 40)  # shifted down from the top edge
    arrow_length = 40  # in pixels
    arrow_head_length = 8
    arrow_head_width = 4
    # Draw +X arrow (red)
    x_arrow_end = (legend_origin[0] + arrow_length, legend_origin[1])
    draw.line([legend_origin, x_arrow_end], fill=(0, 0, 255), width=1)
    x_tip = x_arrow_end
    x_base_left = (x_arrow_end[0] - arrow_head_length, x_arrow_end[1] - arrow_head_width)
    x_base_right = (x_arrow_end[0] - arrow_head_length, x_arrow_end[1] + arrow_head_width)
    draw.polygon([x_tip, x_base_left, x_base_right], fill=(0, 0, 255))
    draw.text((x_arrow_end[0] + 5, x_arrow_end[1] - 5), "X", font=custom_font, fill=(0, 0, 255))
    # Draw +Z arrow (red)
    z_arrow_end = (legend_origin[0], legend_origin[1] + arrow_length)
    draw.line([legend_origin, z_arrow_end], fill=(0, 0, 255), width=1)
    z_tip = z_arrow_end
    z_base_left = (z_arrow_end[0] - arrow_head_width, z_arrow_end[1] - arrow_head_length)
    z_base_right = (z_arrow_end[0] + arrow_head_width, z_arrow_end[1] - arrow_head_length)
    draw.polygon([z_tip, z_base_left, z_base_right], fill=(0, 0, 255))
    draw.text((z_arrow_end[0] + 5, z_arrow_end[1] - 10), "Z", font=custom_font, fill=(0, 0, 255))
    
    # --- Draw oriented triangle marker for the selected point ---
    if selected_point is not None and selected_angle is not None:
        triangle_size = max(8, int(0.015 * sem_map.shape[0]))  
        
        # Adjusted shape for clearer direction indication (longer tip)
        # Scaled-down version (80% of original size)
        scale_factor = 0.8
        local_triangle = np.array([
            [triangle_size * scale_factor, 0],
            [-triangle_size/2 * scale_factor, -triangle_size/3 * scale_factor],
            [-triangle_size/2 * scale_factor, triangle_size/3 * scale_factor]
        ], dtype=np.float32)

        # Rotation logic remains the same
        theta_rad = np.deg2rad(selected_angle)
        R = np.array([
            [np.cos(theta_rad), -np.sin(theta_rad)],
            [np.sin(theta_rad),  np.cos(theta_rad)]
        ], dtype=np.float32)
        rotated_triangle = local_triangle @ R.T
        triangle_pts = rotated_triangle + np.array(selected_point, dtype=np.float32)
        triangle_pts_list = [tuple(pt) for pt in triangle_pts]
        draw.polygon(triangle_pts_list, fill=(0, 0, 255))
    
    # Convert the PIL image back to a numpy array.
    semantic_img = np.array(semantic_pil)
    
    # --- Append palette without altering the semantic map's original size ---
    if with_palette:
        palette_img = get_palette_image()  # This returns a numpy array.
        H = semantic_img.shape[0]
        new_palette_w = int(palette_img.shape[1] * H / palette_img.shape[0])
        palette_img_resized = cv2.resize(palette_img, (new_palette_w, H))
        semantic_img = np.concatenate([semantic_img, palette_img_resized], axis=1)
    
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
    return (world_x, floor_y+0.88, world_z)

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

random.seed(48)

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

    TMP_SAVE_DIR = "tmp/map_image"
    os.makedirs(TMP_SAVE_DIR, exist_ok=True)

    random.shuffle(available_scenes)

    for scene_name in available_scenes:
        scene_sem_map_path =  os.path.join(SEM_MAP_SAVE_ROOT, f"{scene_name}.h5")
        scene_glb_path = os.path.join(SCENE_DIR, f'{scene_name}.glb')
        scene_navmesh_path =  os.path.join(SCENE_DIR, f'{scene_name}.navmesh')

        # Load Habitat config
        sim, action_names, sim_settings = init_sim(scene_glb_path)

        if os.path.exists(scene_navmesh_path):
            sim.pathfinder.load_nav_mesh(scene_navmesh_path)
            print(f"Using NavMesh: {scene_navmesh_path}")

        # Initialize the agent
        agent = sim.get_agent(0)

        with h5py.File(scene_sem_map_path, 'r') as fp:
            floor_ids = sorted([key for key in fp.keys() if is_int(key)])
            for floor_id in floor_ids:
                name = f'{scene_name}_{floor_id}'
                map_world_shift = maps_info[scene_name]['map_world_shift']
                map_y = maps_info[scene_name][floor_id]['y_min']
                resolution = maps_info[scene_name]['resolution']
                map_semantic = np.array(fp[floor_id]['map_semantic'])

                tmp_scene_save_dir = os.path.join(TMP_SAVE_DIR, name)
                os.makedirs(tmp_scene_save_dir, exist_ok=True)

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
                
                for _ in range(3): # get 100 points
                    sampled_pixel = random.choice(nav_coords_px)
                    
                    # Sample a heading from multiples of 30 degrees.
                    sampled_angle = random.choice(range(0, 360, 30))
                    print(f"{name} sampled navigable pixel {sampled_pixel} with direction {sampled_angle}°")
                    
                    sample_save_dir = os.path.join(tmp_scene_save_dir, f"location_{sampled_pixel}_direction_{sampled_angle}")
                    os.makedirs(sample_save_dir, exist_ok=True)

                    map_img_with_sampled_point = visualize_sem_map(map_semantic, selected_point=sampled_pixel, selected_angle=sampled_angle)
                    cv2.imwrite(f"{sample_save_dir}/topdown_map.png", map_img_with_sampled_point)
                    
                    # --- try get the corresponding position in habitat sim and extract image
                    sampled_world_coord = pixel_to_world(sampled_pixel, resolution, np.array(map_world_shift), map_y)
                    print(f"{name} sampled navigable pixel {sampled_pixel} -> world position {sampled_world_coord}")


                    agent_state = agent.get_state()

                    habitat_angle = -(sampled_angle+90)%360

                    sampled_quat = common_utils.quat_from_angle_axis(
                        np.deg2rad(habitat_angle), 
                        np.array([0, 1, 0])
                    )
                    agent_state.position = np.array(sampled_world_coord)
                    agent_state.rotation = sampled_quat
                    agent.set_state(agent_state)
                    print("Agent placed at:", agent.get_state().position)
                    print("Agent rotation (quaternion):", agent.get_state().rotation)

                    # --- Capture surrounding images from that position ---
                    rgb_images = []
                    for rel_angle in [0, 90, 180, 270]:
                        if rel_angle != 0:
                            for _ in range(3):
                                action = "turn_left"
                                print("action", action)
                                sim.step(action)

                        # # Compute the absolute angle (map & simulation) for this view.
                        abs_angle = (sampled_angle + rel_angle) % 360
                        # habitat_angle = -(abs_angle+90)%360
                        # agent_state = agent.get_state()

                        # rotation = common_utils.quat_from_angle_axis(np.deg2rad(habitat_angle), np.array([0, 1, 0]))
                        # agent_state.rotation = rotation
                        # agent.set_state(agent_state)
                        
                        obs = sim.get_sensor_observations()
                        rgb_img = obs["rgb"]
                        rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                        rgb_images.append(rgb_bgr)
                        
                        cv2.imwrite(f"{sample_save_dir}/View_{abs_angle:.0f}.png", rgb_bgr)
                        print(f"Captured view at absolute angle: {abs_angle:.0f}°")
                    
                    panorama = np.hstack(rgb_images)
                    cv2.imwrite(f"{sample_save_dir}/Panorama.png", panorama)

        sim.close()
        break
