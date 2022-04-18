import torch
# To export
# model = torch.hub.load('pytorch/vision:v0.9.0', 'deeplabv3_resnet101', pretrained=True).eval()
# torch.save(model, 'DeepLab.pth')

# To load
# model = torch.jit.load('DeepLab.pth').eval().to(device)


import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.multiprocessing as multiprocessing


def run_deeplab(torch_input, model, pytorch_device, dtype):
	nn_input = torch_input.to(pytorch_device, dtype)
	nn_output = model(nn_input[None, ...])
	
	nn_output = nn_output.cpu().detach().numpy()
	
	return nn_output

class PersonMask:
	def __init__(self):
		pass;
		
	def initialize(self):
		# Setup the V2VModel
		model_path = "src/DeepLab.pt"
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
















import numpy as np
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose_estimator = mp_pose.Pose(
		static_image_mode=False,
		model_complexity=2,
		enable_segmentation=True,
		min_detection_confidence=0.5) 

def create_mask(rgb):
	results = pose_estimator.process(rgb)
	
	mask = results.segmentation_mask
	
	if mask is not None:
		mask = mask.astype(np.uint8, copy=True)
		mask[mask > 0.1] = 1
		mask[mask < 1] = 0
		mask = mask
	return mask
