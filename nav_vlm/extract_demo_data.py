import os
import argparse
import json
import numpy as np
import cv2
from PIL import Image
from tqdm import tqdm, trange
import collections

import msgpack
import msgpack_numpy
msgpack_numpy.patch()

from habitat_baselines.config.default import get_config
from habitat_baselines.common.environments import NavRLEnv

import torch
import torchvision.transforms as T
import math
from copy import deepcopy

from offline_bc.utils.topdown_map import(
    SEM_MAP_SAVE_ROOT,
    get_agent_current_floor_id,
    get_map_loc_dir_from_sim,
    extract_sem_map_patch,
    visualize_sem_map,
)
import h5py
from tqdm import tqdm
from habitat_sim.utils.common import colorize_ids
from itertools import accumulate

# Define action maps.
ACTION_MAPS = {
    'STOP': 0, 'MOVE_FORWARD': 1, 'TURN_LEFT': 2, 'TURN_RIGHT': 3, 'LOOK_UP': 4, 'LOOK_DOWN': 5
}
ID_ACTION_MAPS = {v: k for k, v in ACTION_MAPS.items()}
def get_action_txt(action_id):
    return ID_ACTION_MAPS[action_id]

# Category mappings for semantic labels.
category_to_task_category_id = {
    'chair': 0, 'table': 1, 'picture': 2, 'cabinet': 3, 'cushion': 4,
    'sofa': 5, 'bed': 6, 'chest_of_drawers': 7, 'plant': 8, 'sink': 9,
    'toilet': 10, 'stool': 11, 'towel': 12, 'tv_monitor': 13, 'shower': 14,
    'bathtub': 15, 'counter': 16, 'fireplace': 17, 'gym_equipment': 18,
    'seating': 19, 'clothes': 20
}
category_to_mp3d_category_id = {
    'chair': 3, 'table': 5, 'picture': 6, 'cabinet': 7, 'cushion': 8,
    'sofa': 10, 'bed': 11, 'chest_of_drawers': 13, 'plant': 14, 'sink': 15,
    'toilet': 18, 'stool': 19, 'towel': 20, 'tv_monitor': 22, 'shower': 23,
    'bathtub': 25, 'counter': 26, 'fireplace': 27, 'gym_equipment': 33,
    'seating': 34, 'clothes': 38
}
# Build LABEL_MAP: MP3D label -> task label.
LABEL_MAP = {v: category_to_task_category_id[k] for k, v in category_to_mp3d_category_id.items()}
LABEL_MAP = {}
for k, v in category_to_mp3d_category_id.items():
    LABEL_MAP[v] = category_to_task_category_id[k]

# add cameras to capture surrounding images
def equip_surrounding_views(sim_config, n_views=4, with_depth=True, with_sem=True):
    ########### add surrounding views ##############
    if n_views == 4:
        hfov = 90
        orientations = [
            [0, 0, 0],                # Front
            [0, math.pi / 2, 0],       # Right
            [0, math.pi, 0],          # Back
            [0, 3 / 2 * math.pi, 0],   # Left
        ]
        sensor_dir=[
            "front", 
            "right", "back", "left"]
        sensor_uuids = []
        for camera_id in range(n_views):
            camera_template = f"RGB_{sensor_dir[camera_id]}"
            camera_config = deepcopy(sim_config.TASK_CONFIG.SIMULATOR.RGB_SENSOR)
            camera_config.HFOV = hfov
            camera_config.ORIENTATION = orientations[camera_id]
            camera_config.UUID = camera_template.lower()
            sensor_uuids.append(camera_config.UUID)
            setattr(sim_config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
            sim_config.SENSORS.append(camera_template)
            sim_config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
            # Depth
            if with_depth:
                camera_template = f"DEPTH_{sensor_dir[camera_id]}"
                camera_config = deepcopy(sim_config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR)
                camera_config.HFOV = hfov
                camera_config.ORIENTATION = orientations[camera_id]
                camera_config.UUID = camera_template.lower()
                sensor_uuids.append(camera_config.UUID)
                setattr(sim_config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                sim_config.SENSORS.append(camera_template)
                sim_config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
            # Semantic
            if with_sem:
                camera_template = f"SEMANTIC_{sensor_dir[camera_id]}"
                camera_config = deepcopy(sim_config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR)
                camera_config.HFOV = hfov
                camera_config.ORIENTATION = orientations[camera_id]
                camera_config.UUID = camera_template.lower()
                sensor_uuids.append(camera_config.UUID)
                setattr(sim_config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                sim_config.SENSORS.append(camera_template)
                sim_config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
    elif n_views == 3:
        hfov = 120
        orientations = [
            [0, 0, 0],                  # Front (0°)
            [0, 2 * math.pi / 3, 0],      # Right (120°)
            [0, 4 * math.pi / 3, 0],      # Left (240°)
        ]
        sensor_dir = [
            "front", 
            "right", "left"]
        sensor_uuids = []
        for camera_id in range(n_views):
            # RGB
            camera_template = f"RGB_{sensor_dir[camera_id]}"
            camera_config = deepcopy(sim_config.TASK_CONFIG.SIMULATOR.RGB_SENSOR)
            camera_config.HFOV = hfov
            camera_config.ORIENTATION = orientations[camera_id]
            camera_config.UUID = camera_template.lower()
            sensor_uuids.append(camera_config.UUID)
            setattr(sim_config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
            sim_config.SENSORS.append(camera_template)
            sim_config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
            # Depth
            if with_depth:
                camera_template = f"DEPTH_{sensor_dir[camera_id]}"
                camera_config = deepcopy(sim_config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR)
                camera_config.HFOV = hfov
                camera_config.ORIENTATION = orientations[camera_id]
                camera_config.UUID = camera_template.lower()
                sensor_uuids.append(camera_config.UUID)
                setattr(sim_config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                sim_config.SENSORS.append(camera_template)
                sim_config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
            # Semantic
            if with_sem:
                camera_template = f"SEMANTIC_{sensor_dir[camera_id]}"
                camera_config = deepcopy(sim_config.TASK_CONFIG.SIMULATOR.SEMANTIC_SENSOR)
                camera_config.HFOV = hfov
                camera_config.ORIENTATION = orientations[camera_id]
                camera_config.UUID = camera_template.lower()
                sensor_uuids.append(camera_config.UUID)
                setattr(sim_config.TASK_CONFIG.SIMULATOR, camera_template, camera_config)
                sim_config.SENSORS.append(camera_template)
                sim_config.TASK_CONFIG.SIMULATOR.AGENT_0.SENSORS.append(camera_template)
    ##############################################################

# combine images to panoram
def create_rgb_panorama(right, front, left, back=None, blend_width=2):
    H, W, _ = front.shape
    
    if back is None:
        # 3-view stitching.
        panorama = np.concatenate([right, front, left], axis=1).astype(float)
        # Blend at seams: at columns W and 2W.
        for seam in [W, 2 * W]:
            for i in range(blend_width):
                alpha = i / blend_width
                left_idx = seam - blend_width + i
                right_idx = seam + i
                panorama[:, left_idx, :] = (1 - alpha) * panorama[:, left_idx, :] + alpha * panorama[:, right_idx, :]
        return np.clip(panorama, 0, 255).astype(np.uint8)
    else:
        # 4-view stitching.
        total_width = 4 * W
        # Reorder into [right, front, left, back]
        ordered = [right, front, left, back]
        panorama = np.concatenate(ordered, axis=1).astype(float)
        # Blend seams at boundaries: columns at W, 2W, and 3W.
        for seam in [W, 2 * W, 3 * W]:
            for i in range(blend_width):
                alpha = i / blend_width
                left_idx = seam - blend_width + i
                right_idx = seam + i
                panorama[:, left_idx, :] = (1 - alpha) * panorama[:, left_idx, :] + alpha * panorama[:, right_idx, :]
        # Perform circular shift: front image originally occupies columns [W, 2W] (center at 1.5W).
        # We shift by 0.5W so that the front view center moves to total_width/2 = 2W.
        shift = int(0.5 * W)
        shifted = np.concatenate([panorama[:, -shift:], panorama[:, :-shift]], axis=1)
        return np.clip(shifted, 0, 255).astype(np.uint8)

def create_depth_panorama(right_depth, front_depth, left_depth, back_depth=None, blend_width=2):
    right_depth = np.squeeze(right_depth)
    front_depth = np.squeeze(front_depth)
    left_depth  = np.squeeze(left_depth)
    H, W = front_depth.shape
    
    if back_depth is None:
        panorama = np.concatenate([right_depth, front_depth, left_depth], axis=1)
        for seam in [W, 2 * W]:
            for i in range(blend_width):
                alpha = i / blend_width
                left_idx = seam - blend_width + i
                right_idx = seam + i
                panorama[:, left_idx] = (1 - alpha) * panorama[:, left_idx] + alpha * panorama[:, right_idx]
        return panorama
    else:
        back_depth = np.squeeze(back_depth)
        total_width = 4 * W
        panorama = np.zeros((H, total_width), dtype=float)
        panorama[:, 0:W]     = right_depth
        panorama[:, W:2*W]   = front_depth
        panorama[:, 2*W:3*W] = left_depth
        panorama[:, 3*W:4*W] = back_depth
        for seam in [W, 2 * W, 3 * W]:
            for i in range(blend_width):
                alpha = i / blend_width
                left_idx = seam - blend_width + i
                right_idx = seam + i
                panorama[:, left_idx] = (1 - alpha) * panorama[:, left_idx] + alpha * panorama[:, right_idx]
        shift = int(0.5 * W)
        shifted = np.concatenate([panorama[:, -shift:], panorama[:, :-shift]], axis=1)
        return shifted
    
def create_semantic_panorama(right, front, left, back=None):
    right = np.squeeze(right)
    front = np.squeeze(front)
    left = np.squeeze(left)
    
    if back is None:
        panorama = np.concatenate([right, front, left], axis=1)
    else:
        back = np.squeeze(back)
        panorama = np.concatenate([right, front, left, back], axis=1)
    return panorama

def get_panorama_from_obs(observations):
    panorama_rgb = create_rgb_panorama(
            observations['rgb_right'], 
            observations['rgb_front'], 
            observations['rgb_left'],
            # observations['rgb_back']
        )

    panorama_depth = create_depth_panorama(
        observations['depth_right'],
        observations['depth_front'],
        observations['depth_left'],
        # observations['depth_back']
    )

    panorama_sem = create_semantic_panorama(
        observations['semantic_right'],
        observations['semantic_front'],
        observations['semantic_left'],
        # observations['semantic_back']
    )
    return panorama_rgb, panorama_depth, panorama_sem

# get env config
def get_sim_config(args):
    config = get_config(args.cfg_file)
    config.defrost()

    cfg = config.TASK_CONFIG
    cfg.DATASET.CONTENT_SCENES = [args.scene_id]

    cfg.DATASET.MAX_REPLAY_STEPS = 2000
    cfg.ENVIRONMENT.MAX_EPISODE_STEPS = 2000

    cfg.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = False
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
    cfg.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1

    orientation = np.deg2rad(args.camera_init_pitch)
    cfg.SIMULATOR.DEPTH_SENSOR.HFOV = args.camera_hfov
    cfg.SIMULATOR.DEPTH_SENSOR.POSITION = [0, args.camera_height, 0]
    cfg.SIMULATOR.DEPTH_SENSOR.ORIENTATION = [orientation, 0, 0]
    cfg.SIMULATOR.RGB_SENSOR.HFOV = args.camera_hfov
    cfg.SIMULATOR.RGB_SENSOR.POSITION = [0, args.camera_height, 0]
    cfg.SIMULATOR.RGB_SENSOR.ORIENTATION = [orientation, 0, 0]
    cfg.SIMULATOR.SEMANTIC_SENSOR.HFOV = args.camera_hfov
    cfg.SIMULATOR.SEMANTIC_SENSOR.POSITION = [0, args.camera_height, 0]
    cfg.SIMULATOR.SEMANTIC_SENSOR.ORIENTATION = [orientation, 0, 0]
    cfg.SIMULATOR.AGENT_0.SENSORS.append('SEMANTIC_SENSOR')
    
    # change to set size directly
    cfg.SIMULATOR.SEMANTIC_SENSOR.WIDTH = args.image_width
    cfg.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = args.image_height
    cfg.SIMULATOR.DEPTH_SENSOR.WIDTH = args.image_width
    cfg.SIMULATOR.DEPTH_SENSOR.HEIGHT = args.image_height
    cfg.SIMULATOR.RGB_SENSOR.WIDTH = args.image_width
    cfg.SIMULATOR.RGB_SENSOR.HEIGHT = args.image_height

    # equip surrounding camera
    equip_surrounding_views(config, args.num_views)
    config.freeze()
    return config

# extract demo data
def extract_demo_obs_and_fts(args):
    # build env
    config = get_sim_config(args)

    env = NavRLEnv(config)
    num_episodes = len(env.episodes)
    print('num episodes:', num_episodes)

    # build encoders
    torch.set_grad_enabled(False)
    
    if args.save_topdown_map:
        maps_info = json.load(open(os.path.join(SEM_MAP_SAVE_ROOT, 'semmap_GT_info.json')))
        floor_id = get_agent_current_floor_id(env.habitat_env.sim)
        scene_name = args.scene_id
        scene_sem_map_path =  os.path.join(SEM_MAP_SAVE_ROOT, f"{scene_name}.h5")
        map_world_shift = maps_info[scene_name]['map_world_shift']
        map_resolution = maps_info[scene_name]['resolution']
        with h5py.File(scene_sem_map_path, "r") as fp:
            # map_y = maps_info[scene_name][floor_id]['y_min']
            map_semantic = np.array(fp[floor_id]['map_semantic'])

    # Create scene-specific output directories.
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)
    scene_dir = os.path.join(outdir, args.scene_id)
    rgb_dir = os.path.join(scene_dir, "rgb")
    depth_dir = os.path.join(scene_dir, "depth")
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)
    # sem_dir = os.path.join(scene_dir, "semantic")
    # os.makedirs(sem_dir, exist_ok=True)
    ann_path = os.path.join(scene_dir, "annotations.jsonl")
    
    for _ in trange(num_episodes):
        observations = env.reset()
        episode_id = env.current_episode.episode_id
        demo_actions = [
            x.action if x.action is not None else 'STOP' \
                for x in env.current_episode.reference_replay[1:]
        ]
        demo_actions = [ACTION_MAPS[x] for x in demo_actions]

        if args.remove_look_actions:
            demo_actions = [x for x in demo_actions if x not in (4, 5) ]

        episode_obs = {'reward': [], 'info': []}
        
        if args.save_topdown_map:
            # initial state in map
            episode_obs.setdefault("map_pos", [])
            episode_obs.setdefault("map_dir", [])

        for action in demo_actions:
            for k, v in observations.items():
                episode_obs.setdefault(k, [])
                episode_obs[k].append(v)
            
            if args.save_topdown_map:
                map_pos, map_dir = get_map_loc_dir_from_sim(env.habitat_env.sim, map_resolution, map_world_shift)
                episode_obs['map_pos'].append(map_pos)
                episode_obs['map_dir'].append(map_dir)

            if args.if_debug:
                panorama_rgb, panorama_depth, panorama_sem = get_panorama_from_obs(observations)
                Image.fromarray(panorama_rgb).save("/home/marmot/Boyang/MEM-Nav/tmp/panorama_rgb.png")
                Image.fromarray((panorama_depth * 255).astype(np.uint8)).save("/home/marmot/Boyang/MEM-Nav/tmp/panorama_depth.png") 
                Image.fromarray(colorize_ids(panorama_sem)).save("/home/marmot/Boyang/MEM-Nav/tmp/panorama_semantic.png")
                cv2.imwrite(f"/home/marmot/Boyang/MEM-Nav/tmp/global_topdown_map.png", visualize_sem_map(map_semantic, map_pos, map_dir))
                
            observations, reward, done, info = env.step(action=action)                
            episode_obs['reward'].append(reward)
            episode_obs['info'].append(info)
            if action == 'STOP':
                break
        
        print(f"Original Demo actions: {len(demo_actions)}")
        # remove collision steps
        if not args.keep_collision_steps:
            print("Removing collision ids")
            collide_step_ids = set()
            gps = episode_obs['gps']
            compass = episode_obs['compass']
            for t in range(len(gps)-1):
                # look up and down also result in the same gps and compass
                if all(np.isclose(gps[t], gps[t+1])) and all(np.isclose(compass[t], compass[t+1])) and demo_actions[t] in [1, 2, 3]:
                    collide_step_ids.add(t)
            
            print(f"Collision actions: {len(collide_step_ids)}")
            # filter unwanted index
            for k, v in episode_obs.items():
                episode_obs[k] = [v[idx] for idx in range(len(v)) if idx not in collide_step_ids]
            demo_actions = [x for idx, x in enumerate(demo_actions) if idx not in collide_step_ids]
        

        print("Removing redundant turning ids")
        filtered_action = []
        for idx, act in enumerate(demo_actions):
            if len(filtered_action) == 0:
                filtered_action.append([idx, act])
            else:
                if act in [0, 1, 4, 5]:
                    filtered_action.append([idx, act])
                else:
                    last_idx, last_act = filtered_action[-1]
                    if last_act == act:
                        filtered_action.append([idx, act])
                    elif last_act in [0, 1, 4, 5]:
                        filtered_action.append([idx, act])
                    else:
                        filtered_action.pop()
        keep_step_ids = [i for i, a in filtered_action]
        print(f"redundant turning actions: {len(demo_actions)-len(keep_step_ids)}")
        # filter unwanted index
        for k, v in episode_obs.items():
            episode_obs[k] = [v[idx] for idx in range(len(v)) if idx in keep_step_ids]
        demo_actions = [x for idx, x in enumerate(demo_actions) if idx in keep_step_ids]
        
        print(f"Final demo actions: {len(demo_actions)}")    
        
        # consistent same actions together
        print(f"Compressing continuous same actions") 
        episode_obs['reward'] = list(accumulate(episode_obs['reward']))
        compressed_actions = []
        keep_step_ids = []
        for idx, act in enumerate(demo_actions):
            if idx == 0:
                keep_step_ids.append(idx)
                compressed_actions.append([act, 1])
            else:
                last_act, last_act_num = compressed_actions[-1]
                if act == last_act:
                    compressed_actions[-1][1]+=1
                else:
                    compressed_actions.append([act, 1])
                    keep_step_ids.append(idx)
        
        print(f"Compressed actions: {len(compressed_actions)}")
        
        for k, v in episode_obs.items():
            episode_obs[k] = [v[idx] for idx in range(len(v)) if idx in keep_step_ids]
        
        # get instant reward for accumulate actions
        minus_reward = [0] + episode_obs['reward'][:-1]
        for i in range(len(compressed_actions)):
            episode_obs['reward'][i] = episode_obs['reward'][i] - minus_reward[i]
        episode_obs['demonstration'] = [(get_action_txt(act), num) for act, num in compressed_actions]
        
        # prepare saving data
        episode_obs['panorama_rgb'] = []
        episode_obs['panorama_depth'] = []
        episode_obs['panorama_sem'] = []
        for i in range(len(compressed_actions)):
            panorama_rgb = create_rgb_panorama(episode_obs['rgb_right'][i], episode_obs['rgb_front'][i], episode_obs['rgb_left'][i])
            panorama_depth = create_depth_panorama(episode_obs['depth_right'][i], episode_obs['depth_front'][i], episode_obs['depth_left'][i])
            panorama_sem = create_semantic_panorama(episode_obs['semantic_right'][i], episode_obs['semantic_front'][i], episode_obs['semantic_left'][i])
            episode_obs['panorama_rgb'].append(panorama_rgb)
            episode_obs['panorama_depth'].append(panorama_depth)
            episode_obs['panorama_sem'].append(panorama_sem)

        for key in ['rgb_front', 'rgb_left', 'rgb_right',
                    'depth_front', 'depth_left', 'depth_right',
                    'semantic_front', 'semantic_left', 'semantic_right',
                    'inflection_weight']:
            if key in episode_obs:
                del episode_obs[key]
        
        if args.save_semantic_fts:
            panorama_sem_fts = np.zeros((len(episode_obs['panorama_sem']), len(LABEL_MAP)), dtype=np.float32)
            for t, x in enumerate(episode_obs['panorama_sem']):
                x = x.flatten()
                label_counter = collections.Counter(x)
                npixels = len(x)
                for label, count in label_counter.most_common():
                    if label in LABEL_MAP:
                        panorama_sem_fts[t, LABEL_MAP[label]] = count / npixels

            episode_obs['panorama_sem_fts'] = panorama_sem_fts 
        
        # save data
        num_steps = len(episode_obs["demonstration"])
        rgb_paths = []
        depth_paths = []
        sem_paths = []
        for i in range(num_steps):
            # Save RGB panorama.
            rgb_filename = f"{episode_id}_{i}.png"
            rgb_path = os.path.join(rgb_dir, rgb_filename)
            Image.fromarray(episode_obs["panorama_rgb"][i]).save(rgb_path)
            rgb_paths.append(rgb_path)
            
            # Save Depth panorama.
            depth_filename = f"{episode_id}_{i}.png"
            depth_path = os.path.join(depth_dir, depth_filename)
            depth_img = Image.fromarray((episode_obs["panorama_depth"][i] * 255).clip(0,255).astype(np.uint8))
            depth_img.save(depth_path)
            depth_paths.append(depth_path)
            
            # # Save Semantic panorama.
            # sem_filename = f"{episode_id}_{i}.png"
            # sem_path = os.path.join(sem_dir, sem_filename)
            # Image.fromarray(colorize_ids(episode_obs["panorama_sem"][i])).save(sem_path)
            # sem_paths.append(sem_path)
        
        # Build meta info.
        meta_info = {
            "episode_id": episode_id,
            "objectgoal": episode_obs["objectgoal"][0].item(),
            "object_category": env.current_episode.object_category,
            "compass": np.stack(episode_obs["compass"], 0).tolist(),
            "gps": np.stack(episode_obs["gps"], 0).tolist(),
            "demonstration": episode_obs.get("demonstration", []),
            "reward": np.array(episode_obs["reward"]).tolist(),
            "info": episode_obs["info"],
            "map_pos": episode_obs.get("map_pos", []),
            "map_dir": episode_obs.get("map_dir", []),
            "rgb_paths": rgb_paths,
            "depth_paths": depth_paths,
            "panorama_sem_fts": np.array(episode_obs["panorama_sem_fts"]).tolist(),
        }
        with open(ann_path, "a") as f:
            f.write(json.dumps(meta_info) + "\n")
        print(f"Saved episode {episode_id} data (all {num_steps} steps) under scene {args.scene_id}.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--if_debug', type=int, default=0)
    parser.add_argument(
        '--cfg_file', help='habitat_baselines config file', 
        default='habitat_baselines/config/objectnav/il_ddp_objectnav.yaml'
    )
    parser.add_argument('--scene_id', help='scene id in mp3d', default='gZ6f7yhEvPG')

    parser.add_argument('--num_views', type=int, default=3, help='degree')
    parser.add_argument('--camera_hfov', type=int, default=79, help='degree')
    parser.add_argument('--camera_height', type=float, default=0.88, help='meter')
    parser.add_argument('--camera_init_pitch', type=int, default=0,
                        help='+lookup, -lookdown (degrees)')
    # parser.add_argument('--agent_height', type=float, default=0.88)
    # parser.add_argument('--agent_radius', type=float, default=0.18)

    parser.add_argument('--save_rgb', action='store_true', default=False)
    parser.add_argument('--save_semantic', action='store_true', default=False)
    parser.add_argument('--save_semantic_fts', action='store_true', default=False)
    parser.add_argument('--save_topdown_map', action='store_true', default=False)
    parser.add_argument('--save_rgb_panorama', action='store_true', default=False)
    parser.add_argument('--save_depth_panorama', action='store_true', default=False)

    parser.add_argument('--image_height', type=int, default=224)
    parser.add_argument('--image_width', type=int, default=224)

    parser.add_argument('--keep_collision_steps', action='store_true', default=False)
    parser.add_argument('--remove_look_actions', type=int, default=1)

    parser.add_argument('--outdir', help='output directory')
    
    args = parser.parse_args()

    if args.scene_id == "all":
        scene_ids = sorted([
            "17DRP5sb8fy", "7y3sRwLe3Va", "cV4RVeZvu5T", "GdvgFV5R1Z5", "kEZ7cmS4wCh", "qoiz87JEwZ2", "sT4fr6TAbpF", "VLzqgDo317F",
            "1LXtFkjw3qL", "82sE5b5pLXE", "D7G3Y4RVNrH", "gZ6f7yhEvPG", "mJXqzFtmKg4", "r1Q1Z4BcV1o", "ULsKaCPVFJR", "VVfe2KiqLaN",
            "1pXnuDYAj8r", "8WUmhLawc2A", "D7N2EKCX4Sj", "HxpKQynjfin", "p5wJjkQkbXX", "r47D5H71a5s", "uNb9QFRL6hY", "Vvot9Ly1tCj",
            "29hnd4uzFmX", "aayBHfsNo7d", "dhjEzFoUFzH", "i5noydFURQK", "Pm6F8kyY3z2", "rPc6DW4iMge", "ur6pFq6Qu1A", "vyrNrziPKCB",
            "5LpN3gDmAk7", "ac26ZMwG7aT", "E9uDoFAP3SH", "JeFG25nYj2p", "pRbA3pwrgk9", "s8pcmisQ38h", "Uxmj2M2itWa", "XcA2TqTSSAj",
            "5q7pvUzZiYa", "B6ByNegPMKs", "e9zR4mvMWw7", "JF19kD82Mey", "PuKPg4mmafe", "S9hNv5qa7GM", "V2XKFyX4ASd", "YmJkqBEsHnH",
            "759xd9YjKW5", "b8cTxDM8gDG", "EDJbREhghzL", "jh4fc5c5qoQ", "PX4nDJXEHrG", "sKLMLpTHeUy", "VFuaQ6m2Qom", "ZMojNkEp431"
        ])

        print(f"Found {len(scene_ids)} scenes:")
        for sid in tqdm(scene_ids, desc=f"Processing Available demo scenes"):
            args.scene_id = sid
            print(sid)
            extract_demo_obs_and_fts(args)
    else:
        extract_demo_obs_and_fts(args)

if __name__ == '__main__':
    main()
