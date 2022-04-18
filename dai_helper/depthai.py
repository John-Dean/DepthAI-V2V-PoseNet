import depthai as dai
import numpy as np

def get_device_parameters(device):
	device_calibration_data = device.readCalibration()
	
	depth_queue = device.getOutputQueue("depth", 1, blocking=False)
	
	sample_image	=	depth_queue.get();

	width	=	int(sample_image.getWidth() / 2);
	height	=	int(sample_image.getHeight() / 2)
	
	intrinsics	=	np.array(device_calibration_data.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, width, height))
	extrinsics	=	np.array([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.]])
	
	
	return intrinsics, extrinsics, width, height

