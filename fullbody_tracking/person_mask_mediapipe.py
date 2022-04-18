import numpy as np
import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

pose_estimator = mp_pose.Pose(
		static_image_mode=False,
		model_complexity=2,
		enable_segmentation=True,
		min_detection_confidence=0.5) 

def create_mask(rgb, rotation):
	total_rotation = 0;
	rotated_input = rgb;
	
	if rotation == 90 or rotation == 180 or rotation == 270:
		total_rotation = rotation / 90;
		rotated_input = np.rot90(rgb, k=total_rotation)
	
	# cv2.imshow("pre", rgb)
	# cv2.waitKey(1)
	# cv2.imshow("post", rotated_input)
	# cv2.waitKey(1)
	results = pose_estimator.process(rotated_input)
		
	mask = results.segmentation_mask
	
	if mask is not None:
		mask = np.copy(mask);
		mask[mask > 0.1] = 255
		mask[mask < 1] = 0
		mask = mask.astype(np.uint8, copy=True)
		
		condition = np.stack((results.segmentation_mask,) * 3, axis=-1) > 0.1
		bg_image = np.zeros(rotated_input.shape, dtype=np.uint8)
		bg_image[:] = (192, 192, 192)
		annotated_image = rotated_input
		annotated_image = np.where(condition, annotated_image, bg_image)
		cv2.imshow("post_mask", annotated_image)
		cv2.waitKey(1)
		
		if total_rotation > 0:
			total_rotation = 4 - total_rotation;
			mask = np.rot90(mask, k=total_rotation)
		
		
		# grayImage = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
		# cv2.imshow("output_mask", grayImage)
		# cv2.waitKey(1)
		
	return mask
