import os
import numpy as np
from scipy.spatial import distance_matrix
from torch.utils.data import Dataset
import torch
from blessings import Terminal
from vedo.mesh import Mesh
from data_io import Data_IO
import time
from imeshsegnet import knn

class Mesh_Dataset(Dataset):
    'Data generator'
    def __init__(self, from_docker = False, arch = 'lower', is_train_data = True, train_split = 0.75, patch_size = 6000, model_use="meshsegnet"):
        self.arch = arch
        self.patch_size = patch_size
        self.progress_count = 0
        self.epoch = 0
        self.total_epoch = 0
        self.is_train_data = is_train_data
        self.data_source = Data_IO(from_docker, arch)
        self.elapsed = 0
        self.start_time = 0
        self.orders = []
        self.train_split = train_split
        orders = self.data_source.orders
        split_index = int(len(orders) * train_split)
        if self.is_train_data:
            self.orders = orders[:split_index]
            #self.orders = orders[:10] #fast testing
        else:
            self.orders = orders[split_index:]
            #self.orders = orders[10:12]#fast testing
        #statistics

        self.running_loss = 0
        self.running_mdsc = 0
        self.running_msen = 0
        self.running_mppv = 0
        self.running_batch = 0
        self.total_batches = 0
        self.downsampling = False


        self.model_use = model_use
    
    def set_data_path(self, new_path):
        """Set data_path manually and rescan that path to update orders
        """
        if os.path.exists(new_path):
            self.data_source.set_data_path(new_path)
            orders = self.data_source.orders
            split_index = int(len(orders) * self.train_split)
            if self.is_train_data:
                self.orders = orders[:split_index]
                #self.orders = orders[:10] #fast testing
            else:
                self.orders = orders[split_index:]
                #self.orders = orders[10:12]#fast testing
            #statistics

    
    def __len__(self):
        return len(self.orders)

    def print_progress(self, ordernum):
        total = len(self.orders)
        self.progress_count +=1        
        if (self.progress_count > total):
            self.progress_count = 1
        if self.progress_count == 1:
            self.elapsed = 0
            self.start_time = time.perf_counter()
        else:
            self.elapsed = time.perf_counter() - self.start_time
        percent = self.progress_count / total
        term = Terminal()
        pos = -1
        if self.is_train_data:
            pos = -2
        with term.location(0, term.height + pos):
            #sys.stdout.write(f"Progress: {(percent * 100):.2f}%    ")
            msg = ''
            if self.is_train_data:
                msg = f"Epoch: {self.epoch}/{self.total_epoch}. {self.elapsed:.2f} segs - Training on {ordernum} => {self.progress_count}/{total}. loss: {self.running_loss:.4f}, dsc: {self.running_mdsc:.4f}, sen: {self.running_msen:.4f}, ppv: {self.running_mppv:.4f}"
            else:
                msg = f"Epoch: {self.epoch}/{self.total_epoch}. {self.elapsed:.2f} segs - Validating on {ordernum} => {self.progress_count}/{total}. loss: {self.running_loss:.4f}, dsc: {self.running_mdsc:.4f}, sen: {self.running_msen:.4f}, ppv: {self.running_mppv:.4f}"
            msg += f" Progress: {(percent * 100):.2f}%    " 
            print(msg, end='\r')

    def __getitem__(self, idx):        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ordernum = self.orders[idx]
        self.print_progress(ordernum)

        #ordernum = 20173982 #20176068

        #print(f"reading: {ordernum}")        

        self.data_source.read_model(ordernum)
               
        mesh =Mesh([self.data_source.vertexs, self.data_source.faces])
        mesh.celldata['labels'] = self.data_source.faces_label
        if self.downsampling:
            return mesh
        labels = mesh.celldata['labels'].astype('int32').reshape(-1, 1)

        #create one-hot map
        #        label_map = np.zeros([mesh.cells.shape[0], self.num_classes], dtype='int32')
        #        label_map = np.eye(self.num_classes)[labels]
        #        label_map = label_map.reshape([len(labels), self.num_classes])

        # move mesh to origin
        cells = np.zeros([mesh.NCells(), 9], dtype='float32')
        points = mesh.points().copy()
        faces = mesh.faces().copy()
        for i in range(len(cells)):
            # cells[i][0], cells[i][1], cells[i][2] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(0)) # don't need to copy
            # cells[i][3], cells[i][4], cells[i][5] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(1)) # don't need to copy
            # cells[i][6], cells[i][7], cells[i][8] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(2)) # don't need to copy
            cells[i][0], cells[i][1], cells[i][2] = points[faces[i][0]]
            cells[i][3], cells[i][4], cells[i][5] = points[faces[i][1]]
            cells[i][6], cells[i][7], cells[i][8] = points[faces[i][2]]

        mean_cell_centers = mesh.centerOfMass()
        cells[:, 0:3] -= mean_cell_centers[0:3]
        cells[:, 3:6] -= mean_cell_centers[0:3]
        cells[:, 6:9] -= mean_cell_centers[0:3]

        # customized normal calculation; the vtk/vedo build-in function will change number of points
        v1 = np.zeros([mesh.NCells(), 3], dtype='float32')
        v2 = np.zeros([mesh.NCells(), 3], dtype='float32')
        v1[:, 0] = cells[:, 0] - cells[:, 3]
        v1[:, 1] = cells[:, 1] - cells[:, 4]
        v1[:, 2] = cells[:, 2] - cells[:, 5]
        v2[:, 0] = cells[:, 6] - cells[:, 3]
        v2[:, 1] = cells[:, 7] - cells[:, 4]
        v2[:, 2] = cells[:, 8] - cells[:, 5]
        mesh_normals = np.cross(v1, v2)
        #np.savetxt("normals0.txt", mesh_normals)
        mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
        mesh_normals[:, 0] /= mesh_normal_length[:]
        mesh_normals[:, 1] /= mesh_normal_length[:]
        mesh_normals[:, 2] /= mesh_normal_length[:]
        mesh.celldata['Normal'] = mesh_normals
        #np.savetxt("normals10.txt", mesh_normals)
        # preprae input and make copies of original data
        points = mesh.points().copy()
        points[:, 0:3] -= mean_cell_centers[0:3]
        normals = mesh.celldata['Normal'].copy() # need to copy, they use the same memory address
        barycenters = mesh.cellCenters() # don't need to copy
        barycenters -= mean_cell_centers[0:3]

        #normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
            cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
            cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
            barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
            normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

        X = np.column_stack((cells, barycenters, normals))
        Y = labels

        

        # output to visualize
        #        mesh2 = Easy_Mesh()
        #        mesh2.cells = X_train[:, 0:9]
        #        mesh2.update_cell_ids_and_points()
        #        mesh2.cell_attributes['Normal'] = X_train[:, 12:15]
        #        mesh2.cell_attributes['Label'] = Y_train
        #        mesh2.to_vtp('tmp.vtp')

        if self.model_use == "meshsegnet":
            # initialize batch of input and label
            X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
            Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')
            S1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
            S2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')

            

            # calculate number of valid cells (tooth instead of gingiva)
            positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
            negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx

            #num_positive = len(positive_idx) # number of selected tooth cells
            
            num_positive = 5000
            #print(num_positive)

            if num_positive > self.patch_size: # all positive_idx in this patch
                positive_selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
                selected_idx = positive_selected_idx
            else:   # patch contains all positive_idx and some negative_idx
                num_negative = self.patch_size - num_positive # number of selected gingiva cells
                positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
                negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
                selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

            selected_idx = np.sort(selected_idx, axis=None)

            X_train[:] = X[selected_idx, :]
            Y_train[:] = Y[selected_idx, :]

            D = distance_matrix(X_train[:, 9:12], X_train[:, 9:12])
            S1[D<0.1] = 1.0
            S1 = S1 / np.dot(np.sum(S1, axis=1, keepdims=True), np.ones((1, self.patch_size)))

            S2[D<0.2] = 1.0
            S2 = S2 / np.dot(np.sum(S2, axis=1, keepdims=True), np.ones((1, self.patch_size)))

            X_train = X_train.transpose(1, 0)
            Y_train = Y_train.transpose(1, 0)

            sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train),
                        'A_S': torch.from_numpy(S1), 'A_L': torch.from_numpy(S2)}
            return sample
        
        # Calculate KNN with values 6 and 12
        transposed_barycenters = barycenters.transpose(1, 0)
        transposed_barycenters = transposed_barycenters.reshape([1, transposed_barycenters.shape[0], transposed_barycenters.shape[1]])
        transposed_barycenters = torch.from_numpy(transposed_barycenters)
        knn_6 = knn(transposed_barycenters, 6)
        knn_12 = knn(transposed_barycenters, 12)

        X_train = X.transpose(1, 0)
        Y_train = X.transpose(1, 0)

        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train),
                'knn_6': knn_6, 'knn_12': knn_12}
        return sample

    def get_data_to_predict(self, mesh, start_index, predict_size):

        points = mesh.points().copy()
        faces = mesh.faces().copy()

        num_cells = start_index + predict_size if mesh.NCells() > start_index + predict_size else mesh.NCells()
        cells_size = predict_size if mesh.NCells() > start_index + predict_size else mesh.NCells() - start_index
        cells = np.zeros([cells_size, 9], dtype='float32')
        for i in range(start_index, num_cells):
            index = i - start_index
            # cells[i][0], cells[i][1], cells[i][2] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(0)) # don't need to copy
            # cells[i][3], cells[i][4], cells[i][5] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(1)) # don't need to copy
            # cells[i][6], cells[i][7], cells[i][8] = mesh.polydata().GetPoint(mesh.polydata().GetCell(i).GetPointId(2)) # don't need to copy
            cells[index][0], cells[index][1], cells[index][2] = points[faces[i][0]]
            cells[index][3], cells[index][4], cells[index][5] = points[faces[i][1]]
            cells[index][6], cells[index][7], cells[index][8] = points[faces[i][2]]

        mean_cell_centers = mesh.centerOfMass()
        cells[:, 0:3] -= mean_cell_centers[0:3]
        cells[:, 3:6] -= mean_cell_centers[0:3]
        cells[:, 6:9] -= mean_cell_centers[0:3]

        #cells_size = predict_size if mesh.NCells() > start_index + predict_size else (start_index + predict_size) - mesh.NCells()

        cells_size = len(cells)

        # customized normal calculation; the vtk/vedo build-in function will change number of points
        v1 = np.zeros([cells_size, 3], dtype='float32')
        v2 = np.zeros([cells_size, 3], dtype='float32')
        v1[:, 0] = cells[:, 0] - cells[:, 3]
        v1[:, 1] = cells[:, 1] - cells[:, 4]
        v1[:, 2] = cells[:, 2] - cells[:, 5]
        v2[:, 0] = cells[:, 6] - cells[:, 3]
        v2[:, 1] = cells[:, 7] - cells[:, 4]
        v2[:, 2] = cells[:, 8] - cells[:, 5]
        mesh_normals = np.cross(v1, v2)
        #np.savetxt("normals0.txt", mesh_normals)
        mesh_normal_length = np.linalg.norm(mesh_normals, axis=1)
        mesh_normals[:, 0] /= mesh_normal_length[:]
        mesh_normals[:, 1] /= mesh_normal_length[:]
        mesh_normals[:, 2] /= mesh_normal_length[:]
        #mesh.celldata['Normal'] = mesh_normals
        #np.savetxt("normals10.txt", mesh_normals)
        # preprae input and make copies of original data
        #points = mesh.points().copy()
        points[:, 0:3] -= mean_cell_centers[0:3]
        #normals = mesh.celldata['Normal'].copy() # need to copy, they use the same memory address
        normals = mesh_normals
        bc = self.__get_cell_center(cells)
        #barycenters = mesh.cellCenters() # don't need to copy
        barycenters = bc
        barycenters -= mean_cell_centers[0:3]

        #normalized data
        maxs = points.max(axis=0)
        mins = points.min(axis=0)
        means = points.mean(axis=0)
        stds = points.std(axis=0)
        nmeans = normals.mean(axis=0)
        nstds = normals.std(axis=0)

        for i in range(3):
            cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
            cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
            cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
            barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
            normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

        X = np.column_stack((cells, barycenters, normals))

        if self.model_use == "meshsegnet":
            # computing A_S and A_L
            A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            D = distance_matrix(X[:, 9:12], X[:, 9:12])
            A_S[D<0.1] = 1.0
            A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            A_L[D<0.2] = 1.0
            A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            # numpy -> torch.tensor
            X = X.transpose(1, 0)
            X = X.reshape([1, X.shape[0], X.shape[1]])
            #X = torch.from_numpy(X).to(device, dtype=torch.float)
            A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
            A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
            #A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
            #A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)
            
            return X, A_S, A_L, cells_size
        
        # Calculate KNN with values 6 and 12
        transposed_barycenters = barycenters.transpose(1, 0)
        transposed_barycenters = transposed_barycenters.reshape([1, transposed_barycenters.shape[0], transposed_barycenters.shape[1]])
        transposed_barycenters = torch.from_numpy(transposed_barycenters)
        knn_6 = knn(transposed_barycenters, 6)
        knn_12 = knn(transposed_barycenters, 12)
        # numpy -> torch.tensor
        X = X.transpose(1, 0)
        X = X.reshape([1, X.shape[0], X.shape[1]])

        return X, knn_6, knn_12, cells_size

    def __get_cell_center(self, cells):
        centers = []
        for cell in cells:
            c1 = cell[:3]
            c2 = cell[3:6]
            c3 = cell[6:9]
            xc = (c1[0] + c2[0] + c3[0])/3
            yc = (c1[1] + c2[1] + c3[1])/3
            zc = (c1[2] + c2[2] + c3[2])/3
            c = [xc, yc, zc]
            centers.append(c)
        return np.array(centers)
