import os
from vedo import show, mag2
import vedo.io as IO
import multiprocessing


class Mesh_Aligner():
	def __init__(self, source_path, index, arch, ordernum):
		#source_path = "/home/osmani/3DScans/"
		dest_path = "/media/osmani/Data/AI-Data/Aligned"
		self.path = os.path.join(source_path, str(ordernum))
		self.output_path = os.path.join(dest_path, str(ordernum))		
		self.TARGET_SIZE = 30000
		self.arch = arch
		self.ordernum = ordernum
		self.index = index		
		#self.arch_model = "Maxillary.stl" if arch=="upper" else "Mandibular.stl"
		self.scan_model = "scan_upper.stl" if arch=="upper" else "scan_lower.stl"
		self.ground_truth_path = "/home/osmani/src/Ground_truth"
		self.dest_model = self.load(os.path.join(self.ground_truth_path, self.scan_model))
		self.orig_model = self.load(os.path.join(self.path, self.scan_model))

		#####################################3DScans
		#self.dest_model = self.load("/home/osmani/3DScans/Ground_truth/Maxillary.stl")
		#self.orig_model = self.load("/home/osmani/3DScans/Ground_truth/scan_upper.stl")
		#######################################3

		self.dest_model_aligned = None
		if self.orig_model is not None:
			os.makedirs(self.output_path, exist_ok = True) 


	def get_reduce_ratio(self, msh):
		ncells = msh.NCells()
		frac = self.TARGET_SIZE * 1. / ncells
		return frac
		

	def reduce(self, msh_):
		msh = msh_.clone()
		ratio = self.get_reduce_ratio(msh)
		msh.decimate(fraction=ratio, method='pro', boundaries=True)
		return msh
		

	def load_and_reduce(self, in_dir):
		_m = IO.load(in_dir)
		reduced_m = self.reduce(_m)
		return _m, reduced_m

	def load(self, in_dir):
		try:
			_m = IO.load(in_dir)
			if len(_m.points()) == 0:
				print(f"Failed to load {in_dir}")
				return None		
			return _m
		except Exception:
			print(f"Can't load {in_dir}")
		return None
	

	def calculate_distance(self, m1, m2):
		d = 0
		n = m2.N()
		for p in m1.points():
			cpt = m2.closestPoint(p)
			d += mag2(p - cpt)
		return d / n

	def align_meshes(self, color = 'blue'):	
		minimun_dist = 6.0	
		output_file_path = os.path.join(self.output_path, self.scan_model)
		if os.path.exists(output_file_path):
			print (f"{output_file_path} already exists. Skipping...")
			return		
		print(F"Aligning {self.ordernum}")
		min_dist = {
			'x': 0,
			'y': 0,
			'z': 0,
			'dst': 1e15
		}
		d = min_dist['dst']
		for x in range(0, 181, 45):
			if d<minimun_dist:
				break
			for y in range(0, 181, 45):
				if d<minimun_dist:
					break
				for z in range(0, 181, 45):
					_m1 = self.orig_model.clone()
					_m1.rotateX(x)
					_m1.rotateY(y)
					_m1.rotateZ(z)
					_m1.alignTo(self.dest_model, rigid=True)
					d = self.calculate_distance(_m1, self.dest_model)
					print(f"{self.index}: Best distance: {min_dist['dst']}\nactual distance: {d}\nrotation angle: ({x}, {y}, {z})\n\n")
					if d < min_dist['dst']:
						min_dist['dst'] = d
						min_dist['x'] = x
						min_dist['y'] = y
						min_dist['z'] = z
					if d<minimun_dist:
						break
		
		_m1 = self.orig_model.clone()
		_m1.rotateX(min_dist['x'])
		_m1.rotateY(min_dist['y'])
		_m1.rotateZ(min_dist['z'])
		_m1.alignTo(self.dest_model, rigid=True)
		_m1.c(color)
		self.dest_model_aligned = _m1
	
	def save_model(self):
		if self.dest_model_aligned == None:
			print("Nothing to save")
			return
		output_file_path = os.path.join(self.output_path, self.scan_model)

		#####################
		#output_file_path = "/home/osmani/3DScans/Ground_truth/scan_upper_aligned.stl"
		#####################
		
		IO.write(self.dest_model_aligned, output_file_path)
	
	def show_models(self):
		if self.dest_model_aligned == None:
			print("Nothing to display. Dest model is Nil")
			return
		show([self.dest_model, self.dest_model_aligned], viewup='z', axes=1, title = '3D view').close()	

def process_order_arch(source_path, index, arch, ordernum):
	print(f"Processing {ordernum} => {arch} in thread {index}")
	failed_load = f"failed_load.{arch}"
	invalid = os.path.join(source_path, str(ordernum), failed_load)
	if (os.path.exists(invalid)):
		print(f"Skiping invalid model in order {ordernum} => {arch}")
		return
	ma = Mesh_Aligner(source_path, index, arch, ordernum)
	if (ma.orig_model == None):		
		open(invalid, 'w').close()
		print(f"Invalid model in order {ordernum} => {arch}")
		return	
	ma.align_meshes()
	ma.save_model()

def thread_function(index, source_path, orders_lst):
	for ordernum in orders_lst:
		process_order_arch(source_path, index, "lower", ordernum)
		process_order_arch(source_path, index, "upper", ordernum)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

if __name__ == '__main__':	
	num_threads = 40
	orders = []
	source_path = "/home/osmani/3DScans/"
	
	#thread_function(1, source_path, [20225096])

	if os.path.exists(source_path):
		orders = [ f.name for f in os.scandir(source_path) if f.is_dir() ] 
	c = int(len(orders) / num_threads)
	orders_lists = chunks(orders, c)	
	print(F"Processing {len(orders)} orders")
	result = []
	index = 0
	for orders_lst in orders_lists:
		index +=1
		process = multiprocessing.Process(target = thread_function, args=(index, source_path, orders_lst))
		process.start()
		result.append(process)			
	print(f"Working {len(result)} Threads")
	# Ensure all of the processes have finished
	for j in result:
		j.join()	
	
	#thread_function("upper", 20168668)