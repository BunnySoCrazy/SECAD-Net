import numpy as np
import os
import torch
import torch.utils.data
import h5py


class GTSamples(torch.utils.data.Dataset):
	"""Dataset for training
	"""
	def __init__(
		self,
		data_source,
		grid_sample=64,
		test_flag=False,
	):
		print('data source', data_source)
		self.data_source = data_source
  
		if test_flag:
			filename_shapes = os.path.join(self.data_source, 'ae_test.hdf5')
			name_file = os.path.join(self.data_source, 'test_names.npz')
			npz_shapes = np.load(name_file)
			self.data_names = npz_shapes['test_names']
		else:
			name_file = os.path.join(self.data_source, 'train_names.npz')
			npz_shapes = np.load(name_file)
			filename_shapes = os.path.join(self.data_source, 'ae_train.hdf5')
			self.data_names = npz_shapes['train_names']

		data_dict = h5py.File(filename_shapes, 'r')
		data_voxels = torch.from_numpy(data_dict['voxels'][:]).float()
		self.data_voxels = data_voxels.squeeze(-1).unsqueeze(1)
		self.data_points = torch.from_numpy(data_dict['points_'+str(grid_sample)][:]).float()

		print('Loaded voxels shape, ', self.data_voxels.shape)
		print('Loaded points shape, ', self.data_points.shape)

	def __len__(self):
		return len(self.data_voxels)

	def __getitem__(self, idx):
		return {"voxels":self.data_voxels[idx], "occ_data":self.data_points[idx]}


class VoxelSamples(torch.utils.data.Dataset):
	"""Dataset for fine-tuning and testing
	"""
	def __init__(
		self,
		data_source
	):
		print('data source', data_source)
		self.data_source = data_source
		print('class Samples from voxels')

		name_file = os.path.join(self.data_source, 'test_names.npz')
		npz_shapes = np.load(name_file)
		self.data_names = npz_shapes['test_names']
  
		filename_voxels = os.path.join(self.data_source, 'voxel2mesh.hdf5')
		data_dict = h5py.File(filename_voxels, 'r')
		data_voxels = torch.from_numpy(data_dict['voxels'][:]).float()

		self.data_voxels = data_voxels.squeeze(-1).unsqueeze(1)
		self.data_points = torch.from_numpy(data_dict['points'][:]).float()
		self.data_points[:, :, :3] = (self.data_points[:, :, :3] + 0.5)/64-0.5
		data_dict.close()
			
		print('Loaded voxels shape, ', self.data_voxels.shape)
		print('Loaded points shape, ', self.data_points.shape)


	def __len__(self):
		return len(self.data_voxels)

	def __getitem__(self, idx):
		return self.data_voxels[idx], self.data_points[idx]