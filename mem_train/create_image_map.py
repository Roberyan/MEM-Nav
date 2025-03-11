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
from tqdm import tqdm
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

import imageio
from sim_utils import display_map, get_map_hablab

# --- Paths and settings ---
SCENE_DIR = "data/scene_datasets/gibson_semantic"
SCENE_CONFIG = os.path.join(SCENE_DIR, "gibson_semantic.scene_dataset_config.json")
SEM_MAP_SAVE_ROOT = "data/semantic_maps/gibson/semantic_maps" 
SCENE_BOUNDS_DIR = "data/semantic_maps/gibson/scene_boundaries"
SAVE_DIR = "data/semantic_maps/gibson/image_map_pairs"
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

def visualize_sem_map(sem_map, selected_point=None, selected_angle=None, with_info=True, with_palette=True):
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

    
    if with_info:
        # --- Draw title text (top-center) ---
        # Load a bold sans-serif font at size 10.
        font_path = font_manager.findfont(font_manager.FontProperties(family="sans-serif", weight="bold"))
        custom_font = ImageFont.truetype(font_path, 10)

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
        arrow_length = 20  # in pixels
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
        triangle_size = max(6, int(0.015 * sem_map.shape[0]))  
        
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

def pixel_to_world(pixel, resolution, world_shift, floor_y):
    """
    Convert a pixel coordinate (x, y) from the map to a 3D world coordinate.
    Here, x corresponds to the horizontal axis and y corresponds to the vertical axis in the map.
    """
    x, y = pixel
    world_x = x * resolution + world_shift[0]
    world_z = y * resolution + world_shift[2]
    return [world_x, floor_y+0.88, world_z]

def world_to_pixel(world_pos, resolution, world_shift):
    """
    Convert a 3D world coordinate (x, y, z) to a 2D pixel coordinate (x, y) in the map.

    Parameters:
        world_pos (list or tuple): (world_x, world_y, world_z) in real-world coordinates.
        resolution (float): The map resolution (meters per pixel).
        world_shift (list or tuple): The world shift offsets [shift_x, shift_y, shift_z].

    Returns:
        pixel (tuple): (pixel_x, pixel_y) in the semantic map.
    """
    world_x, _, world_z = world_pos  # Ignore y since it's only height
    pixel_x = int((world_x - world_shift[0]) / resolution)
    pixel_y = int((world_z - world_shift[2]) / resolution)
    return (pixel_x, pixel_y)
# --- smoothing the sem map ---
from scipy import stats, ndimage
import matplotlib.pyplot as plt
# Define your mode filter, excluding class 0
def mode_filter(values):
    nonzero = values[values != 0]
    if len(nonzero) == 0:
        return 0
    mode_val, _ = stats.mode(nonzero, axis=None, keepdims=True, nan_policy='omit')
    return mode_val[0]

def get_smoothed_map(sem_map, ker_size=2):
    return ndimage.generic_filter(
        sem_map, 
        function=mode_filter, 
        size=ker_size,
        mode='constant', 
        cval=0
    )

def show_semmap_compare(sem_map, smooth_map):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    # Display the original semantic map
    im0 = axes[0].imshow(visualize_sem_map(sem_map, with_info=False, with_palette=False), cmap='viridis', interpolation='nearest')
    axes[0].set_title("Original Semantic Map")
    axes[0].axis('off')

    # Display the smoothed semantic map
    im1 = axes[1].imshow(visualize_sem_map(smooth_map, with_info=False, with_palette=False), cmap='viridis', interpolation='nearest')
    axes[1].set_title("Smoothed Semantic Map")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

# get floor navigable map from sim
def get_nav_map_from_sim(sim, sample_resolution, y_min, y_max, n_height=5):
    heights = np.linspace(y_min, y_max, n_height)
    occupancy_maps = []
    for h in heights:
        occ_map = sim.pathfinder.get_topdown_view(sample_resolution, h)
        occupancy_maps.append(occ_map)       
    occupancy_maps = np.stack(occupancy_maps, axis=0)
    return np.any(occupancy_maps, axis=0)

# get local map
def extract_sem_map_patch(map_semantic, map_pos, window_size=5, pad_value=0):
    x, y = map_pos
    H, W = map_semantic.shape
    half_window = window_size // 2

    # Pad the map to prevent out-of-bounds errors
    padded_map = np.pad(map_semantic, pad_width=half_window, mode='constant', constant_values=pad_value)
    # Shift coordinates due to padding
    x_padded, y_padded = x + half_window, y + half_window
    return padded_map[y_padded - half_window : y_padded + half_window + 1,
                        x_padded - half_window : x_padded + half_window + 1]

random.seed(48)

if __name__ == "__main__":
    grid_size = 0.05 # m
    object_boundary = 1.0 # m

    scene_paths = sorted(
        glob.glob(
            os.path.join(SEM_MAP_SAVE_ROOT, "*.h5"),
            recursive=True,
        )
    )
    available_scenes = [x.split("/")[-1].split('.')[0] for x in scene_paths ]
    
    nav_scenes = ['Allensville', 'Beechwood', 'Benevolence', 'Coffeen', 'Collierville', 'Cosmos', 'Darden', 'Forkland', 'Hanson', 'Hiteman', 'Klickitat', 'Lakeville', 'Leonardo', 'Lindenwood', 'Markleeville', 'Marstons', 'Merom', 'Mifflinburg', 'Newfields', 'Onaga', 'Pinesdale', 'Pomaria', 'Ranchester', 'Shelbyville', 'Stockman', 'Tolstoy', 'Wainscott', 'Wiconisco', 'Woodbine']
    maps_info = json.load(open(osp.join(SEM_MAP_SAVE_ROOT, 'semmap_GT_info.json')))

    dset = "gibson"

    maps = {}
    names = []
    maps_xyz_info = {}

    visibility_size = 3.0

    TMP_SAVE_DIR = "tmp/map_image"
    os.makedirs(TMP_SAVE_DIR, exist_ok=True)

    random.shuffle(available_scenes)

    for scene_name in tqdm(available_scenes, desc="Processing Scenes", unit="scene"):
        scene_sem_map_path =  os.path.join(SEM_MAP_SAVE_ROOT, f"{scene_name}.h5")
        scene_glb_path = os.path.join(SCENE_DIR, f'{scene_name}.glb')
        scene_navmesh_path =  os.path.join(SCENE_DIR, f'{scene_name}.navmesh')
        scene_bounds =  json.load(open(os.path.join(SCENE_BOUNDS_DIR, f"{scene_name}.json")))

        sample_ratio = 0.015 if scene_name in nav_scenes else 0.01

        map_world_shift = maps_info[scene_name]['map_world_shift']
        resolution = maps_info[scene_name]['resolution']

        # t = scene_bounds[scene_name]['sizes'].copy()
        # t.pop(1)
        # local_map_range = int(
        #     (min(5, min(t)//2)//2)/resolution
        # )
        local_map_range = 64

        # Load Habitat config
        sim, action_names, sim_settings = init_sim(scene_glb_path)

        if os.path.exists(scene_navmesh_path):
            sim.pathfinder.load_nav_mesh(scene_navmesh_path)
            print(f"Using NavMesh: {scene_navmesh_path}")

        # Initialize the agent
        agent = sim.get_agent(0)

        with h5py.File(scene_sem_map_path, 'r') as fp:
            floor_ids = sorted([key for key in fp.keys() if is_int(key)])
            while len(floor_ids):
                floor_id = floor_ids.pop(0)
                name = f'{scene_name}_{floor_id}'            
                map_y = maps_info[scene_name][floor_id]['y_min']
                map_semantic = np.array(fp[floor_id]['map_semantic'])

                nuniq = len(np.unique(map_semantic))
                if nuniq < MIN_OBJECTS_THRESH + 3: # too less objects in the scene, skip
                    continue
                # smoothing try
                # map_smoothed = get_smoothed_map(map_semantic)
                # show_semmap_compare(map_semantic, map_smoothed)

                tmp_scene_save_dir = os.path.join(TMP_SAVE_DIR, name)
                os.makedirs(tmp_scene_save_dir, exist_ok=True)

                # # Convert to one-hot if needed (here we use the original sem_map)
                # semmap_oh = convert_maps_to_oh(map_semantic, dset=dset)

                # make simulator align with extracted sem map
                resolution_adjust = [-0.0001, -0.00005, -0.00015, -0.0002, -0.00019, -0.000126]

                for d_reso in resolution_adjust:
                    sim_topdown_map = sim.pathfinder.get_topdown_view(resolution+d_reso, map_y)
                    if sim_topdown_map.shape == map_semantic.shape:
                        break
                assert sim_topdown_map.shape == map_semantic.shape, f"{name}, {map_semantic.shape}, but {sim_topdown_map.shape}"
                # display_map(sim_topdown_map)
                # hablab_topdown_map = get_map_hablab(sim, map_y, resolution)
                # display_map(hablab_topdown_map)
                # cv2.imwrite(f"{tmp_scene_save_dir}/navigable_topdown_map.png", hablab_topdown_map)

                # get the union of navigable area and sample points
                map_y_hi = scene_bounds[name]['yhi']                
                nav_map_by_sim = get_nav_map_from_sim(sim, resolution + d_reso, map_y, map_y_hi)
                nav_map_by_sem = (map_semantic == FLOOR_ID)
                nav_map_combine = np.logical_and(nav_map_by_sim, nav_map_by_sem)# np.any(np.stack([nav_map_by_sim, nav_map_by_sem], axis=0), axis=0)
                y_coords, x_coords = np.where(nav_map_combine)
                nav_coords_px = list(zip(x_coords, y_coords))

                if not nav_coords_px:
                    print(f"No navigable pixels found for {name}")
                    continue
                
                print(f"{len(nav_coords_px)} navigable positions")
                
                n_samples =  int(len(nav_coords_px)*sample_ratio)
                sampled_points = random.sample(nav_coords_px, n_samples)
                # display_map(nav_map_combine, sampled_points)

                for sampled_pos in tqdm(sampled_points, desc=f"Processing Sampled Positions in {name}"):
                    # --- try get the corresponding position in habitat sim and extract image
                    sampled_world_coord = pixel_to_world(sampled_pos, resolution, np.array(map_world_shift), map_y)
                    if not sim.pathfinder.is_navigable(sampled_world_coord):
                        sampled_world_coord = sim.pathfinder.snap_point(sampled_world_coord)
                        sampled_pos = world_to_pixel(sampled_world_coord , resolution, map_world_shift)
                    if not sim.pathfinder.is_navigable(sampled_world_coord):
                        continue

                    print(f"{name} sampled navigable pixel {sampled_pos} -> world position {sampled_world_coord}")
                    
                    # extract local map:
                    local_map = extract_sem_map_patch(map_semantic, sampled_pos, window_size=local_map_range)
                    if set(np.unique(local_map)).issubset({0, 1, 2}):
                        continue

                    # Sample a heading from multiples of 30 degrees.
                    sampled_angle = random.choice(range(0, 360, 30))
                    print(f"{name} sampled navigable pixel {sampled_pos} with direction {sampled_angle}°")
                    
                    sample_save_dir = os.path.join(tmp_scene_save_dir, f"location_{sampled_pos}_direction_{sampled_angle}")
                    os.makedirs(sample_save_dir, exist_ok=True)

                    global_topdown_map = visualize_sem_map(map_semantic, selected_point=sampled_pos, selected_angle=sampled_angle)
                    local_topdown_map = visualize_sem_map(local_map, selected_point=[local_map_range//2]*2, selected_angle=sampled_angle, with_info=False, with_palette=False)
                    cv2.imwrite(f"{sample_save_dir}/global_topdown_map.png", global_topdown_map)
                    cv2.imwrite(f"{sample_save_dir}/local_topdown_map.png", local_topdown_map)

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

                        # action way to change angle
                        if rel_angle != 0:
                            for _ in range(3):
                                action = "turn_left"
                                print("action", action)
                                sim.step(action)

                        # Compute the absolute angle (map & simulation) for this view.
                        abs_angle = (sampled_angle + rel_angle) % 360

                        # # set angle way to change
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
                    
                    with h5py.File(f"{sample_save_dir}/local_data.h5", "w") as f:
                        f.create_dataset("local_map", data=local_map) 
                        f.create_dataset(f"rgb_views", data=rgb_images, compression="gzip") 
                    # panorama = np.hstack(rgb_images)
                    # cv2.imwrite(f"{sample_save_dir}/Panorama.png", panorama)

        sim.close()
