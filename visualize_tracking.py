import open3d as o3d

class VisualizeTracking:
	def __init__(self):
		pass;
		
	def initialize(self):
		line_set = o3d.geometry.LineSet()

		'''
		joint_id_to_name = {
			0: 'Head',        8: 'Torso',
			1: 'Neck',        9: 'R Hip',
			2: 'R Shoulder',  10: 'L Hip',
			3: 'L Shoulder',  11: 'R Knee',
			4: 'R Elbow',     12: 'L Knee',
			5: 'L Elbow',     13: 'R Foot',
			6: 'R Hand',      14: 'L Foot',
			7: 'L Hand',
		}
		'''
		lines = [
			[0, 1],
			[2, 1],
			[3, 1],
			[2, 3],
			[4,2],
			[5,3],
			[6,4],
			[7,5],
			[8,2],
			[8,3],
			[9,8],
			[10,8],
			[10,9],
			[11,9],
			[12,10],
			[13,11],
			[14,12]
		]
		line_set.lines = o3d.utility.Vector2iVector(lines)
		
		self.visualizer = None
		self.line_set = line_set
		
	def run(self, tracking_data):
		if tracking_data is None:
			return;
			
		self.line_set.points = o3d.utility.Vector3dVector(tracking_data)
		self.line_set.paint_uniform_color([0,0,1])
		
		if self.visualizer is None:
			self.visualizer = o3d.visualization.Visualizer()
			self.visualizer.create_window()
			options = self.visualizer.get_render_option()
			options.point_size = 3.0
			self.visualizer.add_geometry(self.line_set)
		else:
			self.visualizer.update_geometry(self.line_set)
			
		if self.visualizer is not None:
			self.visualizer.poll_events()
			self.visualizer.update_renderer()
		
