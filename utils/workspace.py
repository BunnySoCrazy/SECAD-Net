import os
import json

specifications_filename = "specs.json"

def load_experiment_specifications(experiment_directory):
	print("loading specifications of " + experiment_directory)
	filename = os.path.join(experiment_directory, specifications_filename)
	if not os.path.isfile(filename):
		raise Exception(
			"The experiment directory ({}) does not include specifications file "
			+ '"specs.json"'.format(experiment_directory)
		)
	return json.load(open(filename))

def get_model_params_dir(experiment_dir):
	dir = os.path.join(experiment_dir, "ModelParameters")
	if not os.path.isdir(dir):
		os.makedirs(dir)
	return dir

def get_model_params_dir_shapename(experiment_dir, shapename):
	dir = os.path.join(experiment_dir, shapename)

	if not os.path.isdir(dir):
		os.makedirs(dir)

	return dir
