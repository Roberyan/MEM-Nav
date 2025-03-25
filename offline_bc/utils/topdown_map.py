import numpy as np
from PIL import Image, ImageFont, ImageDraw
from matplotlib import font_manager
from matplotlib import pyplot as plt
import math
from habitat_sim.utils.common import quat_rotate_vector
from habitat_sim.geo import FRONT
import cv2
SEM_MAP_SAVE_ROOT = "data/semantic_maps/mp3d/semantic_maps" 

MP3D_COLOR_PALETTE=[1.0, 1.0, 1.0, 0.9, 0.9, 0.9, 0.3, 0.3, 0.3, 0.12156862765550613, 0.46666666865348816, 0.7058823704719543, 0.6823529601097107, 0.7803921699523926, 0.9098039269447327, 1.0, 0.49803921580314636, 0.054901961237192154, 1.0, 0.7333333492279053, 0.47058823704719543, 0.1725490242242813, 0.6274510025978088, 0.1725490242242813, 0.5960784554481506, 0.8745098114013672, 0.5411764979362488, 0.8078431487083435, 0.8588235378265381, 0.6117647290229797, 0.5490196347236633, 0.4274509847164154, 0.1921568661928177, 0.5803921818733215, 0.40392157435417175, 0.7411764860153198, 0.772549033164978, 0.6901960968971252, 0.8352941274642944, 0.5490196347236633, 0.33725491166114807, 0.29411765933036804, 0.7686274647712708, 0.6117647290229797, 0.5803921818733215, 0.8901960849761963, 0.46666666865348816, 0.7607843279838562, 0.9686274528503418, 0.7137255072593689, 0.8235294222831726, 0.49803921580314636, 0.49803921580314636, 0.49803921580314636, 0.7803921699523926, 0.7803921699523926, 0.7803921699523926, 0.7372549176216125, 0.7411764860153198, 0.13333334028720856, 0.8588235378265381, 0.8588235378265381, 0.5529412031173706, 0.09019608050584793, 0.7450980544090271, 0.8117647171020508, 0.6196078658103943, 0.8549019694328308, 0.8980392217636108, 0.2235294133424759, 0.23137255012989044, 0.4745098054409027]

# collect topdown map data
def get_local_map_and_views(sim, global_semmap, resolution, world_shift, local_map_range=64, debug=False):
    pos = sim.get_agent_state().position # cur world pos of agent
    quat = sim.get_agent_state().rotation # cur quat pos of agent
    agent_px = world_to_pixel(pos, resolution, world_shift) # agent pos in map
    agent_dir_habitat = quat_to_heading_degree(quat) # agent dir in world
    agent_dir_map= degree_from_habitat_to_map(agent_dir_habitat) # agent dir in map
    rgb_views, depth_views = get_surrounding_views(sim) # rgb views & depth views
    local_topdown_map = extract_sem_map_patch( global_semmap, agent_px, window_size=local_map_range)
    if debug:
        visualize_surrounding_views(rgb_views, depth_views, start_angle=agent_dir_map, save_path="/home/marmot/Boyang/MEM-Nav/tmp/surrounding_views.png")
        nav_map_sim = sim.pathfinder.get_topdown_view(resolution, pos[1])
        display_map(nav_map_sim, [agent_px], "/home/marmot/Boyang/MEM-Nav/tmp/nav_map_sim.png")
        cv2.imwrite(f"/home/marmot/Boyang/MEM-Nav/tmp/topdown_global_current.png", visualize_sem_map(global_semmap, selected_point=agent_px,selected_angle=agent_dir_map))
        cv2.imwrite(f"/home/marmot/Boyang/MEM-Nav/tmp/topdown_local_current.png", visualize_sem_map(local_topdown_map,selected_point=[32, 32],selected_angle=agent_dir_map))
    return agent_px, agent_dir_map, local_topdown_map, rgb_views, depth_views

# ACTION_MAPS
# 'STOP': 0, 'MOVE_FORWARD': 1, 'TURN_LEFT': 2, 'TURN_RIGHT': 3, 'LOOK_UP': 4, 'LOOK_DOWN': 5
def get_surrounding_views(sim):
    rgb_images = []
    depth_images = []
    for rel_angle in (0, 90, 180, 270):
        if rel_angle != 0:
            for _ in range(3):
                sim.step(3)

        obs = sim.get_sensor_observations()
        rgb_images.append(obs["rgb"][..., :3])
        depth_images.append(obs["depth"])

    return rgb_images, depth_images

# quat to heading from sim
def quat_to_heading_degree(quat) -> int:
    # 1️⃣ Compute continuous yaw [0,360)
    dir_vec = quat_rotate_vector(quat, FRONT)
    yaw = (math.degrees(math.atan2(dir_vec[0], dir_vec[2])) + 360) % 360
    return (int((yaw + 15) // 30) * 30) % 360

def degree_from_habitat_to_map(habitat_dir):
    return (90-habitat_dir) %360

def degree_from_map_to_habitat(map_dir):
    return -(map_dir+90)%360

# find agent's current level
def get_agent_current_floor_id(sim):
    agent_pos = sim.get_agent_state().position
    semantic_scene = sim.semantic_annotations()
    # Find which level’s AABB contains the agent
    current_level = None
    for lvl in semantic_scene.levels:
        center = np.array(lvl.aabb.center)
        half = np.array(lvl.aabb.sizes) / 2.0
        if np.all(agent_pos >= center - half) and np.all(agent_pos <= center + half):
            current_level = lvl.id
            break
    
    assert current_level is not None, "Error, can not tell agent's floor position."

    print(f"Agent is on level {current_level}")
    return current_level

# get floor navigable map from sim
def get_nav_map_from_sim(sim, sample_resolution, y_min, y_max, n_height=5):
    heights = np.linspace(y_min, y_max, n_height)
    occupancy_maps = []
    for h in heights:
        occ_map = sim.pathfinder.get_topdown_view(sample_resolution, h)
        occupancy_maps.append(occ_map)       
    occupancy_maps = np.stack(occupancy_maps, axis=0)
    return np.any(occupancy_maps, axis=0)

# display a topdown map with matplotlib
def display_map(topdown_map, key_points=None, save_path=None):
    plt.figure(figsize=(12, 8))
    ax = plt.subplot(1, 1, 1)
    ax.axis("off")
    plt.imshow(topdown_map)
    # Plot key points (ensure they are always on top)
    if key_points is not None:
        for point in key_points:
            plt.plot(point[0], point[1], marker="o", markersize=3, alpha=0.8, color="red")  # Added color for visibility

    # Save the map if needed
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()  # Prevent display when saving
        print(f"Map saved to: {save_path}")
    else:
        plt.show(block=False)  # Keep display behavior unchanged

# map world coord transfer
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

def pixel_to_world(pixel, resolution, world_shift, floor_y):
    """
    Convert a pixel coordinate (x, y) from the map to a 3D world coordinate.
    Here, x corresponds to the horizontal axis and y corresponds to the vertical axis in the map.
    """
    x, y = pixel
    world_x = x * resolution + world_shift[0]
    world_z = y * resolution + world_shift[2]
    return [world_x, floor_y+0.88, world_z]

# get local map
def extract_sem_map_patch(map_semantic, map_pos, window_size=5, pad_value=0):
    x, y = map_pos
    half_window = window_size // 2

    # Pad the map to prevent out-of-bounds errors
    padded_map = np.pad(map_semantic, pad_width=half_window, mode='constant', constant_values=pad_value)
    # Shift coordinates due to padding
    x_padded, y_padded = x + half_window, y + half_window
    return padded_map[y_padded - half_window : y_padded + half_window + 1, x_padded - half_window : x_padded + half_window + 1]

def visualize_sem_map(sem_map, selected_point=None, selected_angle=None, with_info=True):
    c_map = sem_map.astype(np.int32)
    color_palette = [int(x * 255.0) for x in MP3D_COLOR_PALETTE]
        
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
    
    return semantic_img

def visualize_surrounding_views(
    rgb_views,
    depth_views,
    start_angle: float,
    save_path: str = None,
    max_depth: float = 5.0,
    figsize_per_col=(3, 3)
):
    view_angles = [0, 90, 180, 270]
    N = len(view_angles)
    rows = 1 + (rgb_views is not None) + (depth_views is not None)

    fig, axes = plt.subplots(
        rows, N,
        figsize=(figsize_per_col[0]*N, figsize_per_col[1]*rows),
        constrained_layout=True,
        squeeze=False,
        gridspec_kw={'height_ratios': [0.1] + [1]*(rows-1)}
    )

    # Top row = headings
    for i, rel in enumerate(view_angles):
        ax = axes[0, i]
        ax.axis("off")
        angle = (start_angle + rel) % 360
        ax.text(0.5, 0.5, f"{angle:.0f}°", ha="center", va="center", fontsize=14)

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
        fig.colorbar(im, ax=axes[row, :], fraction=0.02, label="Depth (m)")

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()