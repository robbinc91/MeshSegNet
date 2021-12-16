import pickle
from sys import platform
from pathlib import Path
import os
import json

class Data_IO():
    """Clase para leer y escribir fichero smh para tarbajo con AI
    los datos se guardan en 4 arreglos
    vertexs => todos los vertices del msh
    vertexs_label => las etiquetas correspondientes a cada vertice
    faces => todas las caras del msh
    faces_labels => las etiquetas correspondient a cada cara
    """
    def __init__(self, from_docker, arch):
        print(F"Reading data for {arch} Arch.")
        self.vertexs = []
        self.vertexs_label = []
        self.faces = []
        self.faces_label = []
        self.arch = arch
        arch_path = arch.title()# "Lower"
        self.orders = []
        # if self.arch == 'upper':
        #     arch_path = "Upper"
        if platform == "linux" or platform == "linux2":
            # linux
            if (from_docker):
                self.base_path = "/tmp/src/"
                self.data_path = "/tmp/src/data/"
            else:
                self.base_path = "/media/osmani/WD Blue SN550/src/autosegmentation/mainsrc/"
                #self.data_path = "/media/osmani/WD Blue SN550/RevisedModels/"  + arch_path + "/"
                self.data_path = "/media/osmani/Data/AI-Data/Filtered_Scans/Decimated-10k/" + arch_path + "/"

        #elif platform == "darwin"
            # OS X
        elif platform == "win32":
            # Windows...
            self.base_path = "F:/src/autosegmentation/mainsrc/"
            self.data_path = "E:/yero/mexico/test_stls/Decimated-2k/Lower/"
        #p = Path(self.data_path)
        #self.orders = [f.name for f in p.iterdir() if f.is_dir()]   

        #get all folders in the path, non recursive
        if os.path.exists(self.data_path):
            self.orders = [ f.name for f in os.scandir(self.data_path) if f.is_dir() ] 
    
    def set_data_path(self, path):
        """Set data_path manually and rescan that path to update orders
        """
        if os.path.exists(path):
            self.data_path = path
            #get all folders in the path, non recursive
            self.orders = [ f.name for f in os.scandir(path) if f.is_dir() ]
    
    def read_model(self, ordernum, msh_file = ''):
        self.vertexs = []
        self.vertexs_label = []
        self.faces = []
        self.faces_label = []
        filename = f"{self.arch}_opengr_pointmatcher_result.msh"
        file = f"{self.data_path}{ordernum}/AI_Data/{filename}"  
        #transitioning to remove "AI_Data" folder from path.
        if os.path.exists(file) == False:
            file = f"{self.data_path}{ordernum}/{filename}"  
        if len(msh_file) > 0 and os.path.exists(msh_file):
            file = msh_file
        # reading filename into arrays declared above
        with open(file, 'r') as f:
            line = f.readline()
            line = f.readline()
            split_line = line.split(" ")
            vertex_count = split_line[1]
            count = 0
            while (count < (int)(vertex_count)):
                line = f.readline()
                split_line = line.split(" ")
                self.vertexs.append(list(map(float, split_line[0:3])))
                self.vertexs_label.append(int(split_line[3]))
                count = count + 1
            line = f.readline()
            split_line = line.split(" ")
            face_count = split_line[1]
            count = 0
            while (count < int(face_count)):
                line = f.readline()
                split_line = line.split(" ")
                self.faces.append(list(map(int, split_line[0:3])))
                self.faces_label.append(int(split_line[3]))
                count = count + 1
            line = f.readline()
            f.close()
    
    def dest_model_exists(self, ordernum, file_index = -1):
        filename = f"{self.arch}_opengr_pointmatcher_result.msh"        
        path = f"{self.data_path}{ordernum}_{file_index}/AI_Data"
        if (file_index < 0):
            path = f"{self.data_path}{ordernum}"
        file = f"{path}/{filename}"
        return os.path.exists(file)

    def write_model(self, ordernum, vertexts = None, faces = None, file_index = -1):
        filename = f"{self.arch}_opengr_pointmatcher_result.msh"        
        path = f"{self.data_path}{ordernum}_{file_index}/AI_Data"
        if (file_index < 0):
            path = f"{self.data_path}{ordernum}"
        file = f"{path}/{filename}"
        if os.path.exists(file):
            return
        os.makedirs(path, exist_ok=True)        
        # reading filename into arrays declared above
        points = vertexts if vertexts is not None else self.vertexs
        #points = self.vertexts
        with open(file, 'w') as f:
            line = f"solid {self.arch}\n"
            f.write(line)
            line = f"vertexs {len(points)}\n"
            f.write(line)
            count = 0
            lines = []
            while (count < len(points)):
                point = points[count]
                line = f"{point[0]:0.6f} {point[1]:0.6f} {point[2]:0.6f} {self.vertexs_label[count]}\n"
                lines.append(line)
                #f.write(line)
                count = count + 1
            faces = faces if faces is not None else self.faces
            line = f"faces {len(faces)}\n"
            lines.append(line)
            count = 0
            while (count < len(faces)):
                cell = faces[count]
                line = f"{cell[0]} {cell[1]} {cell[2]} {self.faces_label[count]}\n"
                lines.append(line)
                count = count + 1
            line = f"endsolid {self.arch}\n"
            lines.append(line)
            f.writelines(lines)
            f.close()   

    def write(self,ordernum, data,  file_index = 0):
        filename = f"{self.arch}_opengr_pointmatcher_result.mdl"
        if file_index > 0:
            path = f"{self.data_path}{ordernum}_{file_index}/AI_Data"
        else:
            path = f"{self.data_path}{ordernum}/AI_Data"
        file = f"{path}/{filename}"
        self.__save_dict(data, file)
    
    def read(self,ordernum, file_index = 0):
        filename = f"{self.arch}_opengr_pointmatcher_result.mdl"
        if file_index > 0:
            path = f"{self.data_path}{ordernum}_{file_index}/AI_Data"
        else:
            path = f"{self.data_path}{ordernum}/AI_Data"
        file = f"{path}/{filename}"
        if (os.path.exists(file)):
            return self.__load_dict(file)
        else:
            return None

    def __save_dict(self,di_, filename_):
        with open(filename_, 'wb') as f:
            pickle.dump(di_, f)

    def __load_dict(self, filename_):
        with open(filename_, 'rb') as f:
            ret_di = pickle.load(f)
        return ret_di




    def __middle_point(self,v1,v2):
        xm = (v1[0] + v2[0])/2
        ym = (v1[1] + v2[1])/2
        zm = (v1[2] + v2[2])/2
        return [xm, ym, zm]

    def recalc_faces(self):
        max = 0
        #buscar el color de la cara basado en el color de los vertices que la conforman
        new_faces = []
        new_faces_labels = []
        for index,face in enumerate(self.faces):
            cv1 = self.vertexs_label[face[0]] #color del vertice 1
            cv2 = self.vertexs_label[face[1]] #color del vertice 2        
            cv3 = self.vertexs_label[face[2]] #color del vertice 3
            fcolor = self.faces_label[index]  #color actual de la cara
            if fcolor > max:
                max = fcolor
            if (cv1 == cv2 == cv3): #si el color de los 3 vertices es igual
                fcolor = cv1
                new_faces.append(face)
                new_faces_labels.append(fcolor)
            #elif (cv1==0 or cv2 == 0 or cv3 == 0):
            #    new_faces.append(face)
            #    new_faces_labels.append(0)
            elif (cv1 == cv2 or cv1 == cv3): # si el color del vertice 1 es igual al 2 o al 3
                #fcolor = cv1
                if cv1 == cv2:
                    mp = self.__middle_point(self.vertexs[face[0]], self.vertexs[face[2]])
                    self.vertexs.append(mp)
                    i3 = self.vertexs.index(mp)
                    mp = self.__middle_point(self.vertexs[face[1]], self.vertexs[face[2]])
                    self.vertexs.append(mp)
                    i4 = self.vertexs.index(mp)
                    new_faces.append((face[0], i4, i3))
                    if cv1 == 0:
                        cv1=cv3
                    new_faces_labels.append(cv1)
                    new_faces.append((face[0], face[1], i4))
                    new_faces_labels.append(cv1)
                    new_faces.append((i3, i4, face[2]))
                    new_faces_labels.append(cv3)
                else:
                    mp = self.__middle_point(self.vertexs[face[0]], self.vertexs[face[1]])
                    self.vertexs.append(mp)
                    i3 = self.vertexs.index(mp)
                    mp = self.__middle_point(self.vertexs[face[1]], self.vertexs[face[2]])
                    self.vertexs.append(mp)
                    i4 = self.vertexs.index(mp)
                    new_faces.append((face[0], i3, i4))
                    if cv1 == 0:
                        cv1=cv2
                    new_faces_labels.append(cv1)
                    new_faces.append((face[0], i4, face[2]))
                    new_faces_labels.append(cv1)
                    new_faces.append((i3, face[1], i4))
                    new_faces_labels.append(cv2)
            elif (cv2==cv3):# si el color del verftice 2 es igual al 3
                mp = self.__middle_point(self.vertexs[face[0]], self.vertexs[face[2]])
                self.vertexs.append(mp)
                i3 = self.vertexs.index(mp)
                mp = self.__middle_point(self.vertexs[face[0]], self.vertexs[face[1]])
                self.vertexs.append(mp)
                i4 = self.vertexs.index(mp)
                new_faces.append((face[2], i3, i4))
                if cv3 == 0:
                    cv3 = cv1
                new_faces_labels.append(cv3)
                new_faces.append((face[2], i4, face[1]))
                new_faces_labels.append(cv3)
                new_faces.append((face[0], i4, i3))
                new_faces_labels.append(cv1)
            else: # si todos los vertices tienen colores diferentes, cogemos el del primer vertice (???!!!!! NO REASON)
                fcolor = 0
                new_faces.append(face)
                new_faces_labels.append(fcolor)
            #faces_label[index] = fcolor
        self.faces = new_faces
        self.faces_label = new_faces_labels
        print(max)
        return
    
