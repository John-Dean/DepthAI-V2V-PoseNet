import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as multiprocessing

from src.v2v_model import V2VModel
from src.v2v_util_pointcloud import V2VVoxelization


def run_v2v_posenet(torch_input, model, pytorch_device, dtype):
	nn_input = torch_input.to(pytorch_device, dtype)
	nn_output = model(nn_input[None, ...])
	
	nn_output = nn_output.cpu().detach().numpy()
	
	return nn_output

class PointcloudToPoints:
	def __init__(self):
		pass;
		
	def initialize(self):
		# Setup the V2VModel
		model_path = "src/v2v_model.pt"
		model = torch.load(model_path);
		model.eval()
		
		pytorch_device = None
		if torch.cuda.is_available():
			pytorch_device  =   torch.device('cuda')  
		else:
			pytorch_device  =   torch.device('cpu')

		print(pytorch_device)
		
		model_dtype = torch.float
		model = model.to(pytorch_device, model_dtype)
		
		# Transform
		cubic_size			=	3 #Size of cube (meters) around the center point to crop
		v2v_voxelization	=	V2VVoxelization(cubic_size=cubic_size, original_size=100, augmentation=True)
		voxelize_input		=	v2v_voxelization.voxelize_pointcloud
		get_output			=	v2v_voxelization.evaluate
		
		self.model = model;
		self.pytorch_device = pytorch_device
		self.model_dtype = model_dtype
		self.input = voxelize_input;
		self.output = get_output
		
	def run(self, pointcloud):
		center_point = pointcloud.get_center()
		
		voxelized_points = self.input(pointcloud, center_point)
		torch_input = torch.from_numpy(voxelized_points)
		
		nn_result = run_v2v_posenet(torch_input, self.model, self.pytorch_device, self.model_dtype)
		
		output = self.output(nn_result, [center_point])[0]
		
		return output
