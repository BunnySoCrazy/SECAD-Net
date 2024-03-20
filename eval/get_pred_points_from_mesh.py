import os
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from eval_utils import write_ply_point_normal, get_edge_points, to_tensor
import trimesh


def main(args):
    
    config_path = os.path.join(args.experiment_directory, 'config.json')
    
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
            
    pred_points_out_folder = Path(config['pred_points_folder'])
    pred_points_out_folder.mkdir(exist_ok=True, parents=True)
    
    pred_edge_points_out_folder = Path(config['pred_edge_points_folder'])
    pred_edge_points_out_folder.mkdir(exist_ok=True, parents=True)
    
    with open(config['eval_namelist_path'], "r") as f:
        eval_list = f.readlines()

    eval_list = [item.strip().split("/") for item in eval_list]

    print('=== Getting pred points and pred edge points ===')
    for idx in tqdm(range(len(eval_list))):
        object_name = eval_list[idx][0]
        gt_file = config['pred_meshs_folder_to'] + object_name + '.obj'

        mesh = trimesh.exchange.load.load(gt_file)
        
        try:
            face_normals = np.array(mesh.face_normals)
        except:
            print('normal error')
            continue

        points=[]
        normals=[]
        num_points_needed = 8192
        
        while len(points) < num_points_needed:
            sample_points, sample_face_indices = trimesh.sample.sample_surface_even(mesh, num_points_needed)
            points.extend(sample_points)
            normals.extend(face_normals[sample_face_indices])

        points = np.array(points[:num_points_needed])
        normals = np.array(normals[:num_points_needed] )  
        
        write_ply_point_normal(
        (pred_points_out_folder / f"{object_name}.ply"),  vertices=points, normals=normals)
        
        resmpl_edge_points = get_edge_points(to_tensor(points), to_tensor(normals),0.5)
        write_ply_point_normal(
        (pred_edge_points_out_folder / f"{object_name}.ply"),  resmpl_edge_points)


if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser(
		description="eval"
	)
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True
	)
 
	args = arg_parser.parse_args()

	main(args)