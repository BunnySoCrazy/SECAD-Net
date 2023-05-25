import torch


def init_seeds(seed=0):
    """
    Initialize random seeds for reproducibility.

    Args:
        seed (int): Seed value for random number generation.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
def add_latent(points, latent_codes):
    """
    Add latent codes to points.

    Args:
        points (torch.Tensor): Point coordinates tensor.
        latent_codes (torch.Tensor): Latent codes tensor.

    Returns:
        torch.Tensor: Tensor with latent codes added to points.
    """
    batch_size, num_of_points, dim = points.shape
    points = points.reshape(batch_size * num_of_points, dim)
    latent_codes = latent_codes.unsqueeze(1).repeat(1, num_of_points, 1).reshape(batch_size * num_of_points, -1)
    out = torch.cat([latent_codes, points], 1)
    
    return out

def save_obj_data(filename, vertex, face):
	"""
	Save vertices and faces to an OBJ file.

	Args:
		filename (str): Name of the output file.
		vertex (numpy.ndarray): Vertex coordinates array.
		face (numpy.ndarray): Face indices array.
	"""
	numver = vertex.shape[0]
	numfac = face.shape[0]
	with open(filename, 'w') as f:
		f.write('# %d vertices, %d faces'%(numver, numfac))
		f.write('\n')
		for v in vertex:
			f.write('v %f %f %f' %(v[0], v[1], v[2]))
			f.write('\n')
		for F in face:
			f.write('f %d %d %d' %(F[0]+1, F[1]+1, F[2]+1))
			f.write('\n')