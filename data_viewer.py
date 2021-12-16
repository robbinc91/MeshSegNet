import numpy as np
from numpy.core.numeric import False_
from vedo import buildLUT, Mesh, Points, show, settings
from data_io import Data_IO
from vedo.plotter import Plotter
import os

settings.useDepthPeeling = True # might help with transparencies

class Msh_Viewer(object):
    """Clase para visualizar un msh usando vedo
    """
    def __init__(self, arch):
        self.msh_data = Data_IO(False, arch)
        self.lut = self.__get_colors_lookup_table()
        self.key_pressed = None   
        self.plotter = None     
        self._observers = []
        self.is_valid_scan = 0
        self.valid_btn = None
        self.close_btn = None

    
    def __get_colors_lookup_table(self):
        # build a custom LookUp Table of colors:
        #               value, color, alpha
        lut = buildLUT([
                        (0,'black'),
                        (1,'gray'),
                        (2,'green'),
                        (3,'blue'),
                        (4,'yellow'),
                        (5,'cyan'),
                        (6,'magenta'),
                        (7,'silver'),
                        (8,'red'),
                        (9,'maroon'),
                        (10,'olive'),
                        (11,'lime'),
                        (12,'purple'),
                        (13,'teal'),
                        (14,'navy'),
                        (15,'chocolate'),
                        (16,'pink'),
                        (17,'indigo'),
                        (18,'slategray'),
                        (19,'seagreen'),
                        (20,'khaki'),
                        (21,'orange'),
                        (22,'salmon'),
                        (23,'brown'),
                        (24,'aquamarine'),
                        (25,'skyblue'),
                        (26,'darkviolet'),
                        (27,'orchid'),
                        (28,'sienna'),
                        (29,'steelblue'),
                        (30,'beige'),
                        (31,'darkgreen'),
                        (32,'coral')                        
                        ])
        return lut    
    
    # add a button to the current renderer (e.i. nr1)
    def validButtonfunc(self):   
        if self.is_valid_scan == -1:
            self.close_btn.switch() 
        self.is_valid_scan = 1
        self.valid_btn.switch()

    def finishForNow(self):
        if self.is_valid_scan == 1:
            self.valid_btn.switch()
        self.is_valid_scan = -1
        self.close_btn.switch()
        
    
    def match_faces(self, mesh, mesh2):
        print("")
        points = mesh.points().copy()
        dest_faces = mesh2.faces().copy()
        points2 = mesh2.points().copy()
        celldata = []
        total = len(dest_faces)
        for face in dest_faces:
            p1 = points2[face[0]]
            p2 = points2[face[1]]
            p3 = points2[face[2]]

            i1 = np.where((points ==p1).all(axis=1))[0][0]
            i2 = np.where((points ==p2).all(axis=1))[0][0]
            i3 = np.where((points ==p3).all(axis=1))[0][0]

            l1 = mesh.pointdata['vlabel'][i1]
            l2 = mesh.pointdata['vlabel'][i2]
            l3 = mesh.pointdata['vlabel'][i3]

            if l1 == l2 or l1 == l3:
                index1 = l1
            elif l2 == l3:
                index1 = l2
            elif l1 == l2 and l1 == l3:
                index1 = l1
            else:
                index1=0
            
            celldata.append(index1)
            print(F"Processed {len(celldata)} of {total}", end='\r')
        mesh2.celldata['labels'] = celldata

    def display_mesh_by_faces(self, ordernum, msh_file = '', show_btns = True, reduce=False, reduce_target = 10000, recalface = False):
        self.msh_data.read_model(ordernum, msh_file)
        #################
        if (recalface):
            self.msh_data.recalc_faces()
        #################
        mesh=Mesh([self.msh_data.vertexs,self.msh_data.faces])
        mesh.celldata['labels'] = self.msh_data.faces_label        
        mesh_data = mesh.celldata['labels']
        mesh.cmap(self.lut, mesh_data, on='cells')
        self.plotter = Plotter(pos = (2400,100), size = (1500, 1000), interactive = True)
        if show_btns:
            self.valid_btn = self.plotter.addButton(
                self.validButtonfunc,
                pos=(0.7, 0.9),  # x,y fraction from bottom left corner
                states=["Is valid?", "Valid model"],
                c=["w", "w"],
                bc=["dg", "dv"],  # colors of states
                font="courier",   # arial, courier, times
                size=25,
                bold=False,
                italic=False,
            )    
            self.close_btn = self.plotter.addButton(
                self.finishForNow,
                pos=(0.7, 0.05),  # x,y fraction from bottom left corner
                states=["Stop?", "Stopping"],
                c=["w", "w"],
                bc=["dr", "db"],  # colors of states
                font="courier",   # arial, courier, times
                size=25,
                bold=False,
                italic=False,
            )
        if reduce:
            num_cells = mesh.NCells()            
            if num_cells > reduce_target:
                print(f"Reducing model.\noriginal number of cells: {num_cells}")
                print(f'Reducing to {reduce_target} cells')            
                mesh.pointdata['vlabel'] = self.msh_data.vertexs_label
                ratio = reduce_target/mesh.NCells() # calculate ratio
                mesh_d = mesh.clone()
                mesh_d.decimate(fraction=ratio, N= reduce_target,method='pro', boundaries=True)            
                mesh_2 = mesh_d.clone()  
                num_cells = mesh_2.NCells()  
                print(f"new number of cells: {num_cells}")
                self.match_faces(mesh, mesh_2)
                mesh = mesh_2.clone()
        if self.msh_data.arch == 'upper':
            self.plotter.show(mesh, camera={'pos':(0,-160,0), 'viewup' : (0,0,1)}, axes=2, interactive = True, title = f"{ordernum} : {self.msh_data.arch}").close()
        else:
            self.plotter.show(mesh, camera={'pos':(0,160,0), 'viewup' : (0,0,1)}, axes=2, interactive = True, title = f"{ordernum} : {self.msh_data.arch}").close()
        

        #show(mesh, pos = (2400,100), viewup='z', axes=1, title=str(ordernum)).close()
    
    def display_mesh_by_vertexs(self, ordernum, path=""):
        if (len(path) > 0):
            self.msh_data.set_data_path(path)
        self.msh_data.read_model(ordernum)

        mesh=Mesh([self.msh_data.vertexs,self.msh_data.faces])        
        mesh.pointdata['labels'] = self.msh_data.vertexs_label
        data = mesh.pointdata['labels']
        mesh.cmap(self.lut, data, on='points')
        show(mesh, viewup='z', axes=1).close()
        #show(mesh, viewup='z', axes=1, title=order).close()
    
    def display_mesh_by_pointcloud(self, ordernum, path=""):
        if (len(path) > 0):
            self.msh_data.set_data_path(path)
        self.msh_data.read_model(ordernum)

        pointCloud=Points(self.msh_data.vertexs)
        pointCloud.pointdata['labels'] = self.msh_data.vertexs_label
        pointCloud.cmap(self.lut,on='points')

        plt = Plotter(pos = (1000,100))
        plt.show(pointCloud, viewup='x', axes=1)
    

if __name__ == '__main__':
    arch = "lower"
    models_base_path = "/media/osmani/Data/AI-Data/Filtered_Scans/Decimated-100k/"
    #models_base_path = "/home/heikel/deb/repo_IA/AIDataTest/"
    ordernum = 20221844
    file = f"{models_base_path}{ordernum}/AI_Data/{arch}_opengr_pointmatcher_result.msh"
    if os.path.exists(file) == False:
            file = f"{models_base_path}{ordernum}/{arch}_opengr_pointmatcher_result.msh"
    viewer = Msh_Viewer(arch)
    
    #viewer.display_mesh_by_faces(ordernum, file, recalface=False)
    #viewer.display_mesh_by_vertexs(ordernum, models_base_path)
    #viewer.display_mesh_by_pointcloud(ordernum, models_base_path)

    models_base_path = f"/media/osmani/Data/AI-Data/Filtered_Scans/Decimated-10k/{arch.title()}/"
    viewer.msh_data.set_data_path(models_base_path)
    for order in viewer.msh_data.orders:
        print(f"{order}")
        
        file = f"{models_base_path}{order}/{arch}_opengr_pointmatcher_result.msh"
        viewer.display_mesh_by_faces(order, file, show_btns = False)
        #viewer.dispaly_mesh_by_faces(order)
    