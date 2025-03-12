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
    FLOOR_ID,
    MIN_OBJECTS_THRESH
)
from envs import init_sim
from sim_utils import display_map, get_map_hablab
import matplotlib.pyplot as plt
# --- Paths and settings ---
from constants import (
    SCENE_DIR,
    SCENE_CONFIG,
    SEM_MAP_SAVE_ROOT,
    SCENE_BOUNDS_DIR
)

SAVE_DIR = "data/semantic_maps/gibson/image_map_pairs"
os.makedirs(SAVE_DIR, exist_ok=True)

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
    is_area_visited
)

# --- smoothing the sem map ---
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


random.seed(1234)

if __name__ == "__main__":
    only_gibson_nav = True
    if_debugg = False

    dset = "gibson"
    nav_scenes = ['Allensville', 'Beechwood', 'Benevolence', 'Coffeen', 'Collierville', 'Cosmos', 'Darden', 'Forkland', 'Hanson', 'Hiteman', 'Klickitat', 'Lakeville', 'Leonardo', 'Lindenwood', 'Markleeville', 'Marstons', 'Merom', 'Mifflinburg', 'Newfields', 'Onaga', 'Pinesdale', 'Pomaria', 'Ranchester', 'Shelbyville', 'Stockman', 'Tolstoy', 'Wainscott', 'Wiconisco', 'Woodbine']
    maps_info = json.load(open(osp.join(SEM_MAP_SAVE_ROOT, 'semmap_GT_info.json')))
    
    if not only_gibson_nav:
        scene_paths = sorted(
            glob.glob(
                os.path.join(SEM_MAP_SAVE_ROOT, "*.h5"),
                recursive=True,
            )
        )
        available_scenes = [x.split("/")[-1].split('.')[0] for x in scene_paths]
    else:
        available_scenes = nav_scenes
    
    local_map_range = 64

    if if_debugg:
        TMP_SAVE_DIR = "tmp/map_image"
        os.makedirs(TMP_SAVE_DIR, exist_ok=True)
        random.shuffle(available_scenes)
    else:
        TMP_SAVE_DIR = SAVE_DIR

    for scene_name in tqdm(available_scenes, desc="Processing Scenes", unit="scene"):
        scene_sem_map_path =  os.path.join(SEM_MAP_SAVE_ROOT, f"{scene_name}.h5")
        scene_glb_path = os.path.join(SCENE_DIR, f'{scene_name}.glb')
        scene_navmesh_path =  os.path.join(SCENE_DIR, f'{scene_name}.navmesh')
        scene_bounds =  json.load(open(os.path.join(SCENE_BOUNDS_DIR, f"{scene_name}.json")))

        sample_ratio = 0.2 if scene_name in nav_scenes else 0.1

        map_world_shift = maps_info[scene_name]['map_world_shift']
        resolution = maps_info[scene_name]['resolution']

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
                
                # add erosion to make point further
                nav_map_combine = binary_erosion(nav_map_combine, structure=np.ones([3]*2))
                
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
                    if is_area_visited(visited_map, sampled_pos, local_map_range // 2):
                        continue  # Skip if already visited

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
                    if sample_ratio == 0.1:
                        mark_visited_area(visited_map, sampled_pos, local_map_range // 6)
                    else:
                        mark_visited_area(visited_map, sampled_pos, local_map_range // 8)
                    if set(np.unique(local_map)).issubset({0, 1, 2}):
                        continue
                    
                    used_points.append(sampled_pos)

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
                                action = "turn_right"
                                # print("action", action)
                                sim.step(action)
                        
                        # allow some time for rendering
                        sim.step("stop")

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
                        rgb_images.append(rgb_img)
                        
                        cv2.imwrite(f"{sample_save_dir}/View_{abs_angle:.0f}.png", rgb_bgr)
                        print(f"Captured view at absolute angle: {abs_angle:.0f}°")      
                    
                    with h5py.File(f"{sample_save_dir}/local_data.h5", "w") as f:
                        f.create_dataset("local_map", data=local_map) 
                        f.create_dataset(f"rgb_views", data=rgb_images, compression="gzip") 
                        f.create_dataset("map_world_shift", data=map_world_shift) 

                    if if_debugg:
                        panorama = np.hstack(rgb_images)
                        cv2.imwrite(f"{sample_save_dir}/Panorama.png", panorama)
                display_map(nav_map_combine, used_points, os.path.join(tmp_scene_save_dir,"sampled_map.png"))
        sim.close()
