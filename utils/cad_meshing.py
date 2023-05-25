import os
import time
import torch
import mcubes
from utils import utils


def create_mesh_mc(
	generator, shape_3d, shape_code, filename, N=128, max_batch=32**3, threshold=0.5
):
	"""
	Create a mesh using the marching cubes algorithm.

	Args:
		generator: The generator of network.
		shape_3d: 3D shape parameters.
		shape_code: Shape code.
		N: Resolution parameter.
		threshold: Marching cubes threshold value.
	"""
	start = time.time()
	mesh_filename = filename
 
	voxel_origin = [0, 0, 0]
	voxel_size = 1

	overall_index = torch.arange(0, N ** 3, 1, out=torch.LongTensor())
	samples = torch.zeros(N ** 3, 4) # x y z sdf cell_num
 
    # Transform the first 3 columns to be the x, y, z index
	samples[:, 2] = overall_index % N
	samples[:, 1] = (overall_index.long() / N) % N
	samples[:, 0] = ((overall_index.long() / N) / N) % N
 
    # Scale the samples to the voxel size and shift by the voxel origin
	samples[:, 0] = (samples[:, 0] * voxel_size) + voxel_origin[2]
	samples[:, 1] = (samples[:, 1] * voxel_size) + voxel_origin[1]
	samples[:, 2] = (samples[:, 2] * voxel_size) + voxel_origin[0]
	samples[:, :3] = (samples[:, :3]+0.5)/N-0.5

	num_samples = N**3

	samples.requires_grad = False

	head = 0
	
	while head < num_samples:
		sample_subset = samples[head : min(head + max_batch, num_samples), 0:3].cuda()

		occ,_,_ = generator(sample_subset.unsqueeze(0), shape_3d, shape_code)
		samples[head : min(head + max_batch, num_samples), 3] = (
			occ.reshape(-1)
			.detach()
			.cpu()
		)
		head += max_batch

	sdf_values = samples[:, 3]
	sdf_values = sdf_values.reshape(N, N, N)
			
	end = time.time()
	print(f"Sampling took: {end - start:.3f} seconds")

	numpy_3d_sdf_tensor = sdf_values.numpy()

	verts, faces = mcubes.marching_cubes(numpy_3d_sdf_tensor, threshold)

	mesh_points = verts
	mesh_points = (mesh_points + 0.5) / N - 0.5
	
	if not os.path.exists(os.path.dirname(mesh_filename)):
		os.makedirs(os.path.dirname(mesh_filename))

	utils.save_obj_data(f"{mesh_filename}.obj", mesh_points, faces)


def create_CAD_mesh(generator, shape_code, shape_3d, CAD_mesh_filepath):
    """
    Reconstruct shapes with sketch-extrude operations.
    
    Notes:
        - This function currently contains no implementation and serves as a stub.
    """
    pass


def draw_2d_im_sketch(shape_code, generator, sk_filepath):
    """
    Draw a 2D sketch.
    
    Notes:
        - This function currently contains no implementation and serves as a stub.
    """
    pass