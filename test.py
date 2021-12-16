from data_io import Data_IO
from datareader import *


data = Mesh_Dataset(from_docker = False, arch = 'lower', is_train_data = True, train_split = 1.0, patch_size = 6000)
#data.set_data_path("/media/osmani/Data/AI-Data/Filtered_Scans/Decimated-50k")
for i in range(0, len(data.orders)):
    a = data[i]
    print(F"{data.orders[i]}")