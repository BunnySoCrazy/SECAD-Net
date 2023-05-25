import argparse
import os
import utils
from utils.workspace import load_experiment_specifications
from trainer import FineTunerAE
from dataset import dataloader

def main(args):
   	# Create experiment directory path
	experiment_directory = os.path.join('./exp_log', args.experiment_directory)
 
 	# Load experiment specifications
	specs = load_experiment_specifications(experiment_directory)

	occ_dataset = dataloader.VoxelSamples(specs["DataSource"])	

	reconstruction_dir = os.path.join(experiment_directory, "Reconstructions")
	MC_dir = os.path.join(reconstruction_dir, 'MC/')  # Dir for marching cube results
	CAD_dir = os.path.join(reconstruction_dir, 'CAD/')  # Dir for sketch-extrude results
	sk_dir = os.path.join(reconstruction_dir, 'sk/')  # Dir for 2d sketch images
 
	for directory in [reconstruction_dir, CAD_dir, sk_dir, MC_dir]:
		if not os.path.isdir(directory):
			os.makedirs(directory)
  
	shape_indexes = list(range(int(args.start), int(args.end)))
	print('Shape indexes all: ', shape_indexes)

	specs["experiment_directory"] = experiment_directory
	ft_agent = FineTunerAE(specs)

	for index in shape_indexes:
		shapename = occ_dataset.data_names[index]
		shape_code, shape_3d = ft_agent.evaluate(shapename, args.checkpoint)

		mesh_filename = os.path.join(MC_dir, shapename)
		CAD_mesh_filepath = os.path.join(CAD_dir, shapename)
		sk_filepath = os.path.join(sk_dir, shapename)

  		# Create CAD mesh
		utils.create_CAD_mesh(ft_agent.generator, shape_code.cuda(), shape_3d.cuda(), CAD_mesh_filepath)

		# Create mesh using marching cubes
		utils.create_mesh_mc(
			ft_agent.generator, shape_3d.cuda(), shape_code.cuda(), mesh_filename, N=int(args.grid_sample), threshold=float(args.mc_threshold)
		)

		# Draw 2D sketch image
		utils.draw_2d_im_sketch(shape_code.cuda(), ft_agent.generator, sk_filepath)


if __name__ == "__main__":

	arg_parser = argparse.ArgumentParser(
		description="test trained model"
	)
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True
	)
	arg_parser.add_argument(
		"--checkpoint",
		"-c",
		dest="checkpoint",
		default="latest"
	)
	arg_parser.add_argument(
		"--start",
		dest="start",
		default=0,
		help="start shape index",
	)
	arg_parser.add_argument(
		"--end",
		dest="end",
		default=1,
		help="end shape index",
	)
	arg_parser.add_argument(
		"--mc_threshold",
		dest="mc_threshold",
		default=0.9,
		help="marching cube threshold",
	)
	arg_parser.add_argument(
		"--gpu",
		"-g",
		dest="gpu",
		required=True,
		help="gpu id",
	)
	arg_parser.add_argument(
		"--grid_sample",
		dest="grid_sample",
		default=128,
		help="sample points resolution option",
	)
	args = arg_parser.parse_args()
 
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
 
	main(args)
	