import os
import sys
import numpy as np
import torch
from datareader import Mesh_Dataset
from meshsegnet import *
import random
from vedo import buildLUT, Mesh, Points, show, settings

def show_mesh(mesh, wintitle = "Prediction"):
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


        data = mesh.celldata['Label']#[:,2]  # pick z-coords, use them as scalar data
        mesh.cmap(lut, data, on='cells')

        # mesh.pointdata['labels'] = vertexs_label
        # data = mesh.pointdata['labels']#[:,2]  # pick z-coords, use them as scalar data
        # mesh.cmap(lut, data, on='points')

        show(mesh, pos=(2100, 100), viewup='z', axes=1, title=wintitle).close()

if __name__ == '__main__':

    model_use = 'meshsegnet'
    #model_use = 'imeshsegnet'

    #gpu_id = utils.get_avail_gpu()
    #gpu_id = 0
    #torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)

    #model_path = './models'
    last_model_name = 'latest_checkpoint.tar'
    best_model_name = 'Arcad_Mesh_Segementation_best.tar'
    
    arch = "upper"
    use_best_model = True

    model_name = best_model_name if use_best_model else last_model_name
    model_msg = "Using best model" if use_best_model else "Using Last Checkpoint model"

    num_classes = 33
    num_channels = 15

    # set model
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

    # load trained model
    model_base_path = f"/home/osmani/src/autosegmentation/TeethSegmentation/models/{arch}/"
    checkpoint = torch.load(os.path.join(model_base_path, model_name), map_location='cpu')
    epoch_init = checkpoint['epoch']
    losses = checkpoint['losses']
    mdsc = checkpoint['mdsc']
    msen = checkpoint['msen']
    mppv = checkpoint['mppv']
    val_losses = checkpoint['val_losses']
    val_mdsc = checkpoint['val_mdsc']
    val_msen = checkpoint['val_msen']
    val_mppv = checkpoint['val_mppv']
    
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device, dtype=torch.float)

    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    #ordernum = 20100029
    # Predicting
    model.eval()

    predict_on_valids = True

    data_path ="/home/osmani/AIData/"
    #data_path = f"/media/osmani/Data/AI-Data/Filtered_Scans/Decimated-100k/{arch.title()}/"
    
    best_dsc = (mdsc[-1] if predict_on_valids else max(val_mdsc)) * 100
    targe_prediction_msg = "Predicting on already seen (train) dataset" if predict_on_valids else "Predicting on never seen before dataset"
    print(f"\n{model_msg}")
    print(f"Model mdsc: {best_dsc:0.2f}%")
    print(f"{targe_prediction_msg}\n")

    data_reader = Mesh_Dataset(from_docker = False, arch = "arch", is_train_data = True, train_split = 1, patch_size = 6000, model_use=model_use)
    orders = [ f.name for f in os.scandir(data_path) if f.is_dir() ]    
    for order in orders:
        #order_to_predict = order
        order_to_predict  = random.choice(orders)
        #order_to_predict = 20176479
        invalid_path = f"{data_path}{order}/AI_Data/invalid.{arch}"
        
        if os.path.exists(invalid_path) == predict_on_valids:
            continue
        stl_path = f"{data_path}{order_to_predict }/AI_Data/{arch}_opengr_pointmatcher_result.msh"
        if not os.path.exists(stl_path):
            stl_path = f"{data_path}{order_to_predict }/{arch}_opengr_pointmatcher_result.msh"
            if not os.path.exists(stl_path):
                continue
    
        with torch.no_grad():
            print(f'Predicting Sample filename: {order_to_predict }')            
            data_reader.data_source.read_model(order_to_predict , msh_file=stl_path)                
            mesh =Mesh([data_reader.data_source.vertexs, data_reader.data_source.faces])
            total_cells = mesh.NCells()
            print(f"Total cells: {total_cells}")    
            target_num = 30000            
            if total_cells > target_num:
                print(f'Downsampling to {target_num} cells...')            
                ratio = target_num/total_cells # calculate ratio
                mesh_d = mesh.clone()
                #mesh_d.decimate(fraction=ratio, method='pro')#, boundaries=True)
                mesh_d.decimate(fraction=ratio, N= target_num,method='pro', boundaries=True)   
                mesh = mesh_d.clone()
                total_cells = mesh.NCells()
                print(f'Mesh reduced to  {total_cells} cells')            
            
            predict_size = 10000       
            predicted_labels = np.empty((0,1), dtype=np.int32)
            for i in range(0, total_cells, predict_size):
                print(i)
                X, term_1, term_2, num_cells = data_reader.get_data_to_predict(mesh, i, predict_size)
            
                predicted_labels_d = np.zeros([num_cells, 1], dtype=np.int32)

                X = torch.from_numpy(X).to(device, dtype=torch.float)
                term_1 = torch.from_numpy(term_1).to(device, dtype=torch.float)
                term_2 = torch.from_numpy(term_2).to(device, dtype=torch.float)
                tensor_prob_output = model(X, term_1, term_2).to(device, dtype=torch.float)
                patch_prob_output = tensor_prob_output.cpu().numpy()

                for i_label in range(num_classes):
                    predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

                predicted_labels = np.append(predicted_labels, predicted_labels_d, axis=0)

                #predicted_labels.append(predicted_labels_d)
                

            # output downsampled predicted labels
            mesh2 = mesh.clone()
            mesh2.celldata['Label'] = predicted_labels
            show_mesh(mesh2, wintitle = f"Prediction on {order_to_predict}")
            # mesh2_path =os.path.join("/media/osmani/WD Blue SN550/RevisedModels/Lower/20168668", 'predicted.vtp')
            # vedo.write(mesh2, mesh2_path)

            # print(f'Sample filename: {mesh2_path} completed')
        
    