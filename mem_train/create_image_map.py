import os
import json
import glob
import random
import h5py
import numpy as np
import cv2
import json
from tqdm import tqdm
import h5py
import glob
import random
import numpy as np
import os.path as osp
import habitat_sim.utils.common as common_utils
from scipy.ndimage import binary_erosion

from constants import (
    SPLIT_SCENES,
    FLOOR_ID,
    MIN_OBJECTS_THRESH
)
from envs import init_sim
from sim_utils import display_map, get_map_hablab

def is_int(s):
    try:
        int(s)
        return True
    except:
        return False


from topdown_map_utils import (
    visualize_sem_map,
    pixel_to_world,
    world_to_pixel,
    get_nav_map_from_sim,
    extract_sem_map_patch,
    mark_visited_area,
    is_area_visited,
    convert_maps_to_oh
)

random.seed(2025)

# align sim topdown map to semantic map size
def resize_map_proportional(target: np.ndarray, ref: np.ndarray) -> np.ndarray:
    h_ref, w_ref = ref.shape
    # Cast to float32 so interpolation produces fractional coverage
    target_f = target.astype(np.float32)
    resized = cv2.resize(
        target_f,
        (w_ref, h_ref),                # cv2 takes (width, height)
        interpolation=cv2.INTER_AREA  # preserves area proportions
    )
    return resized

if __name__ == "__main__":
    dset = "mp3d"
    if "ACTIVE_DATASET" in os.environ:
        dset = os.environ["ACTIVE_DATASET"]  # mp3d / gibson
    
    if_debug = False
    local_map_range = 64 # local map size: local_map_range + 1
    sample_ratio = 0.3 # how many available points to sample
    look_down_p = 0.1
    
    # --- Paths and settings ---
    if dset == 'gibson':
        SCENE_DIR = "data/scene_datasets/gibson_semantic"
        SEM_MAP_SAVE_ROOT = "data/semantic_maps/gibson/semantic_maps" 
        SCENE_BOUNDS_DIR = "data/semantic_maps/gibson/scene_boundaries"
        SAVE_DIR = "data/semantic_maps/gibson/image_map_pairs/train"
        os.makedirs(SAVE_DIR, exist_ok=True)
    elif dset == 'mp3d':
        SCENE_DIR = "data/scene_datasets/mp3d_uncompressed"
        SEM_MAP_SAVE_ROOT = "data/semantic_maps/mp3d/semantic_maps" 
        SCENE_BOUNDS_DIR = "data/semantic_maps/mp3d/scene_boundaries"
        SAVE_DIR = "data/semantic_maps/mp3d/image_map_pairs"
        os.makedirs(SAVE_DIR, exist_ok=True)
    else:
        raise TypeError(f"{dset} is not supported")
    
    only_baseline_scene = False
    
    maps_info = json.load(open(osp.join(SEM_MAP_SAVE_ROOT, 'semmap_GT_info.json')))
    
    if not only_baseline_scene:
        scene_paths = sorted(
            glob.glob(
                os.path.join(SEM_MAP_SAVE_ROOT, "*.h5"),
                recursive=True,
            )
        )
        available_scenes = [x.split("/")[-1].split('.')[0] for x in scene_paths]
    else:
        available_scenes =  SPLIT_SCENES[dset]['train'] + SPLIT_SCENES[dset]['val']

    if if_debug:
        TMP_SAVE_DIR = "tmp/map_image"
        os.makedirs(TMP_SAVE_DIR, exist_ok=True)
        random.shuffle(available_scenes)
    else:
        TMP_SAVE_DIR = SAVE_DIR

    for scene_name in tqdm(available_scenes, desc="Processing Scenes", unit="scene"):
        scene_sem_map_path =  os.path.join(SEM_MAP_SAVE_ROOT, f"{scene_name}.h5")
        scene_bounds =  json.load(open(os.path.join(SCENE_BOUNDS_DIR, f"{scene_name}.json")))
        if dset == 'gibson':
            scene_glb_path = os.path.join(SCENE_DIR, f'{scene_name}.glb')
            scene_navmesh_path =  os.path.join(SCENE_DIR, f'{scene_name}.navmesh')
        elif dset == 'mp3d':
            scene_glb_path = os.path.join(SCENE_DIR, f'{scene_name}/{scene_name}.glb')
            scene_navmesh_path =  os.path.join(SCENE_DIR, f'{scene_name}/{scene_name}.navmesh')

        map_world_shift = maps_info[scene_name]['map_world_shift']
        resolution = maps_info[scene_name]['resolution']

        # Load Habitat config
        sim, action_names, sim_settings = init_sim(
            scene_glb_path, 
            use_sem_sensor=False,
            image_height=448,
            image_width=448,
            hfov=79
        )

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


                tmp_scene_save_dir = os.path.join(TMP_SAVE_DIR, name)
                os.makedirs(tmp_scene_save_dir, exist_ok=True)

                # Convert to one-hot if needed (here we use the original sem_map)
                # semmap_oh = convert_maps_to_oh(map_semantic, dset=dset)

                # make simulator align with extracted sem map
                # hablab_topdown_map = get_map_hablab(sim, map_y, resolution)
                # display_map(hablab_topdown_map)
                # cv2.imwrite(f"{tmp_scene_save_dir}/navigable_topdown_map.png", hablab_topdown_map)

                # get the union of navigable area and sample points
                map_y_hi = scene_bounds[name]['yhi']                
                nav_map_by_sim = get_nav_map_from_sim(sim, resolution, map_y, map_y_hi)
                if nav_map_by_sim.shape != map_semantic.shape:
                    nav_map_by_sim = resize_map_proportional(nav_map_by_sim, map_semantic)
                nav_map_by_sem = (map_semantic == FLOOR_ID)
                nav_map_combine = np.logical_and(nav_map_by_sim, nav_map_by_sem)# np.any(np.stack([nav_map_by_sim, nav_map_by_sem], axis=0), axis=0)
                
                # add erosion to make less point further
                nav_map_combine = binary_erosion(nav_map_combine, structure=np.ones([3]*2))
                
                if if_debug:
                    cv2.imwrite(f"{tmp_scene_save_dir}/nav_map_sim.png", (nav_map_by_sim.astype(np.uint8) * 255))
                    cv2.imwrite(f"{tmp_scene_save_dir}/nav_map_sem.png", ( nav_map_by_sem.astype(np.uint8) * 255))
                    cv2.imwrite(f"{tmp_scene_save_dir}/nav_map_combine.png", (nav_map_combine.astype(np.uint8) * 255))
                
                y_coords, x_coords = np.where(nav_map_combine)
                nav_coords_px = list(zip(x_coords, y_coords))

                if not nav_coords_px:
                    print(f"No navigable pixels found for {name}")
                    continue
                
                print(f"{len(nav_coords_px)} navigable positions")
                
                n_samples =  int(len(nav_coords_px)*sample_ratio)
                sampled_points = random.sample(nav_coords_px, n_samples)
                # display_map(nav_map_combine, sampled_points)

                visited_map = np.zeros_like(map_semantic, dtype=bool)

                used_points = [] 
                for sampled_pos in tqdm(sampled_points, desc=f"Processing Sampled Positions in {name}"):
                    if is_area_visited(visited_map, sampled_pos, local_map_range // 4):
                        continue  # Skip if already visited

                    # --- try get the corresponding position in habitat sim and extract image
                    sampled_world_coord = pixel_to_world(sampled_pos, resolution, np.array(map_world_shift), map_y)
                    while not sim.pathfinder.is_navigable(sampled_world_coord):
                        sampled_world_coord = sim.pathfinder.snap_point(sampled_world_coord)
                        sampled_pos = world_to_pixel(sampled_world_coord , resolution, map_world_shift)

                    print(f"{name} sampled navigable pixel {sampled_pos} -> world position {sampled_world_coord}")
                    
                    # extract local map:
                    local_map = extract_sem_map_patch(map_semantic, sampled_pos, window_size=local_map_range)
                    
                    mark_visited_area(visited_map, sampled_pos, local_map_range // 8)
                    
                    if set(np.unique(local_map)).issubset({0, 1, 2}): # no goal object in the map
                        continue
                    
                    used_points.append(sampled_pos)

                    # Sample a heading from multiples of 30 degrees.
                    sampled_angle = random.choice(range(0, 360, 30))
                    print(f"{name} sampled navigable pixel {sampled_pos} with direction {sampled_angle}°")
                    
                    sample_save_dir = os.path.join(tmp_scene_save_dir, f"location_{sampled_pos}_direction_{sampled_angle}")
                    os.makedirs(sample_save_dir, exist_ok=True)

                    global_topdown_map = visualize_sem_map(map_semantic, dset=dset, selected_point=sampled_pos, selected_angle=sampled_angle)
                    local_topdown_map = visualize_sem_map(local_map,  dset=dset, selected_point=[local_map_range//2]*2, selected_angle=sampled_angle, with_info=False, with_palette=False)
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
                    if_look_down = random.random() <= look_down_p
                    if if_look_down:
                        sim.step("look_down")
                    rgb_images = []
                    depth_images = []
                    for rel_angle in [0, 90, 180, 270]:

                        # action way to change angle
                        if rel_angle != 0:
                            for _ in range(3):
                                action = "turn_right"
                                # print("action", action)
                                sim.step(action)
                        
                        # allow some time for rendering
                        sim.step("stop")

                        # Compute the absolute angle (map & simulation) for this view.
                        abs_angle = (sampled_angle + rel_angle) % 360

                        obs = sim.get_sensor_observations()
                        rgb_img = obs["rgb"]
                        depth_img = obs['depth']
                        rgb_images.append(rgb_img[:,:,:3])
                        depth_images.append(depth_img)
                        print(f"Captured view at absolute angle: {abs_angle:.0f}°")      
                        
                        if if_debug:
                            rgb_bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
                            cv2.imwrite(f"{sample_save_dir}/View_{abs_angle:.0f}.png", rgb_bgr)
                            
                            # Depth → 0–255 uint8 + JET colormap PNG
                            depth_u8 = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
                            cv2.imwrite(f"{sample_save_dir}/Depth_{abs_angle:.0f}.png", depth_color)
                    
                    if if_look_down:
                        sim.step("look_up")
                    
                    with h5py.File(f"{sample_save_dir}/local_data.h5", "w") as f:
                        f.create_dataset("local_map", data=local_map) 
                        f.create_dataset(f"rgb_views", data=rgb_images, compression="gzip") 
                        f.create_dataset("depth_views", data=depth_images, compression="gzip")
                        f.create_dataset("map_world_shift", data=map_world_shift) 

                    if if_debug:
                        panorama = np.hstack(rgb_images)
                        cv2.imwrite(f"{sample_save_dir}/Panorama.png", panorama)
                        
                        depth_color_imgs = []
                        for depth in depth_images:
                            # Normalize depth → 0–255 uint8
                            depth_u8 = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                            # Apply JET colormap for visualization
                            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_JET)
                            depth_color_imgs.append(depth_color)

                        # Create panorama by stacking side‑by‑side
                        depth_panorama = np.hstack(depth_color_imgs)
                        cv2.imwrite(f"{sample_save_dir}/Depth_Panorama.png", depth_panorama)
                        
                display_map(nav_map_combine, used_points, os.path.join(tmp_scene_save_dir,"sampled_map.png"))
        sim.close()
        if if_debug:
            break
