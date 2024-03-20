import os
import shutil
import argparse
import json


def copy_files(src_dir, dest_dir, txt_file):
    os.makedirs(dest_dir, exist_ok=True)

    subdir = dest_dir
    os.makedirs(subdir, exist_ok=True)
    
    with open(txt_file, 'r') as f:
        files = [line.strip() for line in f]

    for root, dirs, filenames in os.walk(src_dir):
        for filename in filenames:
            if filename[:5] in (file[:5] for file in files):
                dest_file = os.path.join(subdir, filename)

                shutil.copy(os.path.join(root, filename), dest_file)
                
def main(args):
    
    config_path = os.path.join(args.experiment_directory, 'config.json')
    
    with open(config_path, 'r') as config_file:
        config = json.load(config_file)
               
    copy_files(config['pred_meshs_folder_ori'], 
               config['pred_meshs_folder_to'], 
               config['eval_namelist_path'])


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