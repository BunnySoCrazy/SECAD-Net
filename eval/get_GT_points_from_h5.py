import os
import json
import argparse
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm
from eval_utils import write_ply_point_normal, get_edge_points, to_tensor


def load_data_names(name_file):
    npz_shapes = np.load(name_file)
    return list(npz_shapes['test_names'])

def create_output_directory(out_dir):
    output_folder = Path(out_dir)
    output_folder.mkdir(exist_ok=True, parents=True)
    return output_folder

def read_evaluation_list(eval_namelist_path):
    with open(eval_namelist_path, "r") as f:
        return [line.strip().split("/") for line in f]

def write_point_clouds_to_ply(data, data_names_list, eval_list, output_folder):
    print('=== Getting GT pcl ===')
    for name in tqdm(eval_list):
        object_idx = data_names_list.index(name[0])
        write_ply_point_normal(
            output_folder / f"{name[0]}.ply", data['points'][object_idx]
        )
        
def process_and_write_edge_points(data, data_names_list, eval_list, output_folder):
    print('=== Getting GT edge pcl ===')
    for name in tqdm(eval_list):
        object_idx = data_names_list.index(name[0])
        gt_resmpl_points = get_edge_points(
            to_tensor(data['points'][object_idx][:,:3]),
            to_tensor(data['points'][object_idx][:,3:]),
            0.9
        )
        write_ply_point_normal(output_folder / f"{name[0]}.ply", gt_resmpl_points)
        
def main(args):
    
    config_path = os.path.join(args.experiment_directory, 'config.json')
    
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
        
    data = h5py.File(config['ABC_points2mesh_path'], 'r')
    data_names_list = load_data_names(config['ABC_test_name_path'])
    output_folder = create_output_directory(config['ground_truth_points_folder'])
    output_edge_points_folder = create_output_directory(config['ground_truth_edge_points_folder'])
    eval_list = read_evaluation_list(config['eval_namelist_path'])
    
    write_point_clouds_to_ply(data, data_names_list, eval_list, output_folder)
    process_and_write_edge_points(data, data_names_list, eval_list, output_edge_points_folder)


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