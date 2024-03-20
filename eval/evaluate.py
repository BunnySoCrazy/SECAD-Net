import os
import json
import argparse
from eval_utils import evaluate


def main(args):
    
    config_path = os.path.join(args.experiment_directory, 'config.json')
    
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)

    print('=== Evaluating points ===')
    evaluate(config['eval_namelist_path'], 
             config['pred_points_folder'], 
             config['ground_truth_points_folder'], 
             config['result_folder'])

    print('=== Evaluating edge points ===')
    evaluate(config['eval_namelist_path'], 
             config['pred_edge_points_folder'], 
             config['ground_truth_edge_points_folder'], 
             config['result_edge_folder'])


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