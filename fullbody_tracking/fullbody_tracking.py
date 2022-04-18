import torch.multiprocessing as multiprocessing

import numpy as np
import cv2
import open3d as o3d

from .person_mask_mediapipe import create_mask
from .pointcloud_to_points import PointcloudToPoints
from .depth_to_pointcloud import convert_to_pointcloud


def thread_fn(input_queue, output_queue, intrinsics, extrinsics, width, height, CAMERA_ROTATION):
	# Setup a Open3D camera using the provided camera parameters
	camera_intrinsics	=	o3d.camera.PinholeCameraIntrinsic(
		width,
		height,
		intrinsics[0][0],intrinsics[1][1],intrinsics[0][2],intrinsics[1][2]
	)
	
	# Setup the V2V model	
	pointcloud_to_points = PointcloudToPoints();
	pointcloud_to_points.initialize();
	
	
	# (Optional) Visualization
	pointcloud = o3d.geometry.PointCloud()
	visualizer = None
	
	while True:
		inputs = input_queue.get(block=True)
		output = None;
		rgb, depth = inputs;
		
		rgb = cv2.resize(rgb, dsize=(width, height), interpolation = cv2.INTER_LANCZOS4)
		depth = cv2.resize(depth, dsize=(width, height), interpolation = cv2.INTER_NEAREST)
		
		# Create a mask of the person
		mask = create_mask(rgb, CAMERA_ROTATION);
		
		if mask is not None:
			grayImage = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
			cv2.imshow("output_mask", grayImage)
			cv2.waitKey(1)
			
			kernel = np.ones((5,5), np.uint8)
			mask_erosion = cv2.erode(mask, kernel, iterations=1)
			
			
			grayImage2 = cv2.cvtColor(mask_erosion, cv2.COLOR_GRAY2BGR)
			cv2.imshow("output_mask_eroded", grayImage2)
			cv2.waitKey(1)
			
			# Convert the person into a pointcloud
			frame_pointcloud = convert_to_pointcloud(depth, mask_erosion, camera_intrinsics, extrinsics, CAMERA_ROTATION)
			
			# Run Voxel To Voxel PoseNet on the pointcloud
			output = pointcloud_to_points.run(frame_pointcloud)
			
			
			# (Optional) Visualization
			pointcloud.points = frame_pointcloud.points
			pointcloud.paint_uniform_color([0,0,1])
			if visualizer is None:
				visualizer = o3d.visualization.Visualizer()
				visualizer.create_window()
				visualizer.add_geometry(pointcloud)
				origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5, origin=[0, 0, 0])
				visualizer.add_geometry(origin)
				
				visualizer.poll_events()
				visualizer.update_renderer()
			else:
				visualizer.update_geometry(pointcloud)
				visualizer.poll_events()
				visualizer.update_renderer()
			cv2.imshow("camera", rgb)
			cv2.waitKey(1)
		
		output_queue.put(output)

class FullBodyTracking:
	def __init__(self, intrinsics, extrinsics, width, height, CAMERA_ROTATION):
		self.image_queue	=	multiprocessing.Queue(1);
		self.tracking_queue	=	multiprocessing.Queue(1);
		self.intrinsics	=	intrinsics;
		self.extrinsics	=	extrinsics;
		self.width	=	width;
		self.height	=	height;
		self.CAMERA_ROTATION	=	CAMERA_ROTATION;
		
		thread	=	multiprocessing.Process(target=thread_fn, args=(self.image_queue,self.tracking_queue,self.intrinsics,self.extrinsics, self.width, self.height, self.CAMERA_ROTATION), name="Fullbody Tracking", daemon=True);
		thread.start();
		
		self.thread	=	thread;
		
	def image(self, rgb, depth):
		self.image_queue.put([rgb, depth]);
		
	def get_result(self):
		result = self.tracking_queue.get(True, timeout=None);
		return result;
