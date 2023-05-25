import os
import argparse
from tqdm import tqdm

from utils import init_seeds
from utils.workspace import load_experiment_specifications
from trainer import FineTunerAE
from dataset import dataloader

def main(args):
    # Set random seed
	init_seeds(10)
 
  	# Load experiment specifications
	experiment_directory = os.path.join('./exp_log', args.experiment_directory)
	specs = load_experiment_specifications(experiment_directory)
 	
  	# Create dataset and data loader
	occ_dataset = dataloader.VoxelSamples(specs["DataSource"])	
 
  	# Indices of shapes that need fine-tuning
	shape_indexes = list(range(int(args.start_index), int(args.end_index)))
	print('Indices of shapes that need fine-tuning: ', shape_indexes)

	epoches_each_stage = int(args.epoches)
	stages = [0, 1]
 
	specs["experiment_directory"] = experiment_directory


	for index in shape_indexes:
		print('Fine-tuning shape index', index)
		shapename = occ_dataset.data_names[index]
		occ_data = occ_dataset.data_points[index].unsqueeze(0).cuda()
		voxels = occ_dataset.data_voxels[index].unsqueeze(0).cuda()
		ft_agent = FineTunerAE(specs)

		for phase in stages:
			start_epoch = ft_agent.load_shape_code(phase, voxels, shapename, args.checkpoint)
   
			# start finetuning
			clock = ft_agent.clock
			pbar = tqdm(range(start_epoch, start_epoch + epoches_each_stage))
   
			for e in pbar:
				for i in range(40):
					outputs, out_info = ft_agent.train_func(occ_data)
					pbar.set_description("EPOCH[{}][{}]".format(e, epoches_each_stage))
					clock.tick()
				pbar.set_postfix(out_info)
				ft_agent.save_model_if_best_per_shape(shapename)
				clock.tock()
    
			ft_agent.save_model_parameters_per_shape(shapename,"last_%d.pth"%(phase))

 
if __name__ == "__main__":
	arg_parser = argparse.ArgumentParser()
 
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
		default="best"
	)
	arg_parser.add_argument(
		"--start",
		dest="start_index",
		default=0
	)
	arg_parser.add_argument(
		"--end",
		dest="end_index",
		default=1
	)
	arg_parser.add_argument(
		"--epoches",
		dest="epoches",
		default=50,
	)
	arg_parser.add_argument(
		"--gpu",
		"-g",
		dest="gpu",
		default=0,
	)
 
	args = arg_parser.parse_args()
 
	os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"]="%d"%int(args.gpu)
 
	main(args)
