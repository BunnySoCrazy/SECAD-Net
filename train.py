import os
import argparse
import torch.utils.data as data_utils
from tqdm import tqdm

from utils import init_seeds
from utils.workspace import load_experiment_specifications
from dataset import dataloader
from trainer import TrainerAE

def main(args):
    # Set random seed
	init_seeds()
 
  	# Create experiment directory path
	experiment_directory = os.path.join('./exp_log', args.experiment_directory)
 
 	# Load experiment specifications
	specs = load_experiment_specifications(experiment_directory)

	# Create dataset and data loader
	occ_dataset = dataloader.GTSamples(specs["DataSource"], test_flag=True)
	data_loader = data_utils.DataLoader(
		occ_dataset,
		batch_size=specs["BatchSize"],
		shuffle=True,
		num_workers=4
	)
 
	specs["experiment_directory"] = experiment_directory
 
	tr_agent = TrainerAE(specs)
 
	# Start training
	clock = tr_agent.clock

	for epoch in range(specs["NumEpochs"]):
		# Begin iteration
		pbar = tqdm(data_loader)
		for b, data in enumerate(pbar):
			# Train step
			outputs, out_info = tr_agent.train_func(data)
			pbar.set_description("EPOCH[{}][{}]".format(epoch, b))
			pbar.set_postfix(out_info)
			clock.tick()
   
		# Save model
		if epoch % specs["SaveFrequency"] == 0:
			tr_agent.save_model_parameters(f"{epoch}.pth")
		tr_agent.save_model_if_best()

		clock.tock()


if __name__ == "__main__":
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument(
		"--experiment",
		"-e",
		dest="experiment_directory",
		required=True
	)
	arg_parser.add_argument(
		"--gpu",
		"-g",
		dest="gpu",
		default=0
	)

	args = arg_parser.parse_args()

	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%int(args.gpu)

	main(args)