import depthai as dai

def create_pipeline():
	# Start defining a pipeline
	pipeline = dai.Pipeline()
	color_node = pipeline.create(dai.node.ColorCamera)
	color_node.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
	# Color cam: 1920x1080
	# Mono cam: 640x400
	color_node.setPreviewSize(1280, 720)
	color_node.setIspScale(2,3) # To match 400P mono cameras
	color_node.setBoardSocket(dai.CameraBoardSocket.RGB)
	# color.initialControl.setManualFocus(130)
	
	xout_color = pipeline.create(dai.node.XLinkOut)
	xout_color.setStreamName("color")
	color_node.preview.link(xout_color.input)
	
	# Left mono camera
	left_node = pipeline.create(dai.node.MonoCamera)
	left_node.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	left_node.setBoardSocket(dai.CameraBoardSocket.LEFT)
	# Right mono camera
	right_node = pipeline.create(dai.node.MonoCamera)
	right_node.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
	right_node.setBoardSocket(dai.CameraBoardSocket.RIGHT)

	# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
	depth_node = pipeline.create(dai.node.StereoDepth)
	
	# Create a node that will produce the depth map (using disparity output as it's easier to visualize depth this way)
	depth_node.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
	# Options: MEDIAN_OFF, KERNEL_3x3, KERNEL_5x5, KERNEL_7x7 (default)
	depth_node.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
	depth_node.setSubpixel(True)
	depth_node.setLeftRightCheck(True)
	depth_node.setExtendedDisparity(False)
	config = depth_node.initialConfig.get()
	# config.postProcessing.speckleFilter.enable = False
	# config.postProcessing.speckleFilter.speckleRange = 50
	# config.postProcessing.temporalFilter.enable = True
	config.postProcessing.spatialFilter.enable = True
	config.postProcessing.spatialFilter.holeFillingRadius = 5
	config.postProcessing.spatialFilter.numIterations = 1
	# print(config.postProcessing.spatialFilter.delta)
	# config.postProcessing.spatialFilter.alpha = 0.25
	# config.postProcessing.spatialFilter.delta = 15
	config.postProcessing.thresholdFilter.minRange = 400
	config.postProcessing.thresholdFilter.maxRange = 15000
	config.postProcessing.decimationFilter.decimationFactor = 2

	depth_node.initialConfig.set(config)
	depth_node.initialConfig.setConfidenceThreshold(75)

	
	# stereo.initialConfig.setBilateralFilterSigma(64000)
	depth_node.setDepthAlign(dai.CameraBoardSocket.RGB)
	left_node.out.link(depth_node.left)
	right_node.out.link(depth_node.right)
	
	# Create depth output
	xout_depth = pipeline.create(dai.node.XLinkOut)
	xout_depth.setStreamName("depth")
	depth_node.depth.link(xout_depth.input)
	
	return pipeline
