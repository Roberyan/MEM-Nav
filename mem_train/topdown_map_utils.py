import numpy as np
from PIL import Image, ImageFont, ImageDraw
import cv2
from matplotlib import font_manager


from constants import (
    GIBSON_CATEGORIES,
    GIBSON_COLOR_PALETTE,
    GIBSON_LEGEND_PALETTE,
    NUM_OBJECT_CATEGORIES,
    CAT_OFFSET,
)

# get map view of each surrounding image with correct angle
def generate_map_view_mask(local_map, sampled_angle, fov=90):
    """
    Generates a visibility mask aligned with the agent’s actual orientation.

    Parameters:
        local_map (np.array): The local semantic map (centered around the agent).
        sampled_angle (int): The agent's heading in degrees (0° = East, 90° = South).
        fov (int): The field of view (default: 90°).

    Returns:
        np.array: A binary mask indicating the correctly aligned visible region.
    """
    H, W = local_map.shape
    cx, cy = W // 2, H // 2  # The agent is always at the center

    # Create an empty mask
    mask = np.zeros_like(local_map, dtype=np.uint8)

    # Iterate through all pixels in the local map
    for y in range(H):
        for x in range(W):
            dx, dy = x - cx, y - cy  # Relative position to agent

            # 🔹 Flip y-axis to match the correct coordinate system
            angle = (np.arctan2(dy, dx) * 180 / np.pi) % 360  # Fix direction

            # Define the view range based on sampled_angle
            start_angle = (sampled_angle - fov / 2) % 360
            end_angle = (sampled_angle + fov / 2) % 360

            # Correctly determine whether a pixel is in the FOV
            if start_angle < end_angle:
                if start_angle <= angle <= end_angle:
                    mask[y, x] = 1
            else:  # Handle wraparound cases (e.g., 330° - 30° FOV)
                if angle >= start_angle or angle <= end_angle:
                    mask[y, x] = 1

    return mask

# map to one hot map
def convert_maps_to_oh(semmap, dset="gibson"): # convert sem map to one hot, skip out-of-bound
    ncat = NUM_OBJECT_CATEGORIES[dset]
    semmap_oh = np.zeros((ncat, *semmap.shape), dtype=np.float32)
    for i in range(0, ncat):
        semmap_oh[i] = (semmap == i + CAT_OFFSET).astype(np.float32)
    return semmap_oh

# benefit sampling
def mark_visited_area(visited_map, sampled_pos, local_map_range):
    """
    Marks a local area around a sampled position as visited.

    Parameters:
        visited_map (np.array): A boolean mask tracking sampled areas.
        sampled_pos (tuple): The (x, y) pixel position being sampled.
        local_map_range (int): The size of the local area to mark.

    Returns:
        None (modifies `visited_map` in place)
    """
    H, W = visited_map.shape
    half_range = local_map_range // 2

    # Compute bounds for the local area (ensure they stay within image bounds)
    x_min = max(0, sampled_pos[0] - half_range)
    x_max = min(W, sampled_pos[0] + half_range + 1)
    y_min = max(0, sampled_pos[1] - half_range)
    y_max = min(H, sampled_pos[1] + half_range + 1)

    # Mark the region as visited
    visited_map[y_min:y_max, x_min:x_max] = True

def is_area_visited(visited_map, sampled_pos, local_map_range):
    """
    Checks if a sampled area has already been visited.

    Parameters:
        visited_map (np.array): A boolean mask tracking sampled areas.
        sampled_pos (tuple): The (x, y) pixel position to check.
        local_map_range (int): The size of the local area to check.

    Returns:
        bool: True if the area has been sampled before, False otherwise.
    """
    H, W = visited_map.shape
    half_range = local_map_range // 2

    # Compute bounds for the local area
    x_min = max(0, sampled_pos[0] - half_range)
    x_max = min(W, sampled_pos[0] + half_range + 1)
    y_min = max(0, sampled_pos[1] - half_range)
    y_max = min(H, sampled_pos[1] + half_range + 1)

    # Check if any pixel in the region is already visited
    return np.any(visited_map[y_min:y_max, x_min:x_max])


# map world coord transfer
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
    half_window = window_size // 2

    # Pad the map to prevent out-of-bounds errors
    padded_map = np.pad(map_semantic, pad_width=half_window, mode='constant', constant_values=pad_value)
    # Shift coordinates due to padding
    x_padded, y_padded = x + half_window, y + half_window
    return padded_map[y_padded - half_window : y_padded + half_window + 1, x_padded - half_window : x_padded + half_window + 1]


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