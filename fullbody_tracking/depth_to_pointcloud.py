import numpy as np
import open3d as o3d


def convert_to_pointcloud(depth, mask, intrinsics, extrinsics, CAMERA_ROTATION):
	masked = mask // 255
	person_depth	=	np.multiply(depth, masked)
	person_depth[person_depth < 0.1] = 0
	
	depth_o3d = o3d.geometry.Image(person_depth)
		
	max_depth_meters = 5.0;
	point_cloud_reduction_factor = 1; #Int, 1 is the lowest/least reduction 
	
	pointcloud = o3d.geometry.PointCloud.create_from_depth_image(depth_o3d, intrinsics, extrinsics, 1000.0, max_depth_meters, point_cloud_reduction_factor)
	
	angle = CAMERA_ROTATION;
	angle = angle/180*np.pi
	
	pointcloud.rotate([
		[np.cos(angle)  , -np.sin(angle)    , 0],
		[np.sin(angle)  , np.cos(angle)     , 0],
		[0              , 0                 , 1],
	], [0,0,0])   
	
	return pointcloud
