from fullbody_tracking.fullbody_tracking import FullBodyTracking
import time
import contextlib
import cv2
import numpy as np

import depthai as dai
from dai_helper.pipeline import create_pipeline
from dai_helper.depthai import get_device_parameters
from visualize_tracking import VisualizeTracking

if __name__ == "__main__":
	CAMERA_ROTATION = 90;
	
	visualize_tracking = VisualizeTracking()
	visualize_tracking.initialize()
	
	with contextlib.ExitStack() as stack:
		# Check a depthAI device exists
		dai_available_devices = dai.Device.getAllAvailableDevices()
		if len(dai_available_devices) == 0:
			raise RuntimeError("No devices found!")
		else:
			print("Found", len(dai_available_devices), "devices")
		
		# Prepare a pipeline for DAI use
		dai_pipeline = create_pipeline();
		
		# Setup the device to use the Pipeline
		device = None;
		for available_device in dai_available_devices:
			device = dai.Device(dai_pipeline, available_device)
			break;
		
		print("Connected");
		
		# Setup device intrinsics
		intrinsics, extrinsics, width, height = get_device_parameters(device);
		depth_queue = device.getOutputQueue("depth", 1, blocking=False)
		color_queue	= device.getOutputQueue("color", 1, blocking=False)
		
		# Setup Fullbody Tracking
		tracking	=	FullBodyTracking(intrinsics, extrinsics, width, height, CAMERA_ROTATION);
		
		# Start program
		counter = 0;
		while True:
			start_time = time.time()
			
			# Get input from the camera
			color_data = color_queue.get()
			depth_data = depth_queue.get()
			
			color_frame	=	color_data.getFrame()
			depth_frame	=	depth_data.getFrame()
			
			counter += 1;
			
			if counter > 50:
				# cv2.imshow("color", color_frame)
				# cv2.waitKey(1)
				tracking.image(color_frame, depth_frame);
				tracking_data	=	tracking.get_result();
				visualize_tracking.run(tracking_data)
				# print(tracking_data)
				
			
			
		