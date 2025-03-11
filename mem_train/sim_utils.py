from PIL import Image
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use non-interactive mode to prevent GUI issues
from matplotlib import pyplot as plt
from habitat.utils.visualizations import maps

def display_sample(rgb_obs, semantic_obs=np.array([]), depth_obs=np.array([])):
    from habitat_sim.utils.common import d3_40_colors_rgb

    rgb_img = Image.fromarray(rgb_obs, mode="RGBA")

    arr = [rgb_img]
    titles = ["rgb"]
    if semantic_obs.size != 0:
        semantic_img = Image.new("P", (semantic_obs.shape[1], semantic_obs.shape[0]))
        semantic_img.putpalette(d3_40_colors_rgb.flatten())
        semantic_img.putdata((semantic_obs.flatten() % 40).astype(np.uint8))
        semantic_img = semantic_img.convert("RGBA")
        arr.append(semantic_img)
        titles.append("semantic")

    if depth_obs.size != 0:
        depth_img = Image.fromarray((depth_obs / 10 * 255).astype(np.uint8), mode="L")
        arr.append(depth_img)
        titles.append("depth")

    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)
    plt.show(block=False)

# convert 3d points to 2d topdown coordinates
def convert_points_to_topdown(pathfinder, points, meters_per_pixel):
    points_topdown = []
    bounds = pathfinder.get_bounds()
    for point in points:
        # convert 3D x,z to topdown x,y
        px = (point[0] - bounds[0][0]) / meters_per_pixel
        py = (point[2] - bounds[0][2]) / meters_per_pixel
        points_topdown.append(np.array([px, py]))
    return points_topdown

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

def get_map_hablab(sim, map_y, grid_size):
    hablab_topdown_map = maps.get_topdown_map(
        sim.pathfinder, map_y, meters_per_pixel=grid_size
    )
    recolor_map = np.array(
        [[255, 255, 255], [128, 128, 128], [0, 0, 0]], dtype=np.uint8
    )
    return recolor_map[hablab_topdown_map]