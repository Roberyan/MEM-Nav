import os
import trimesh
from tqdm import tqdm

data_path = "data/scene_datasets/mp3d_uncompressed"
first_directory = os.listdir(data_path)

for second_directory in tqdm(first_directory):
    mesh_directory = os.path.join(data_path, second_directory, 'matterport_mesh')
    scene_directory = os.listdir(mesh_directory)
    for scene in scene_directory:
        obj_file_path = os.path.join(mesh_directory, scene, scene+'.obj')
        glb_file_path = os.path.join(mesh_directory, scene, second_directory+'.glb')
        if os.path.exists(obj_file_path):
            print(obj_file_path)
            scene = trimesh.load(obj_file_path)
            scene.export(glb_file_path)