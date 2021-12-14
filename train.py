import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from datareader import *
from meshsegnet import *
from imeshsegnet import *
from metrics import *
import utils
import pandas as pd
################################################################
#Modelos que he encontrado con problemas a la hora de leerlos durante el entrenamiento:
#20173152 - upper
#20173355 - lower
#175835 -= lower
################################################################


if __name__ == '__main__':
    model_use = 'meshsegnet'
    #model_use = 'imeshsegnet'
    
    
    arch = 'lower'
    log_index=0
    base_path = "/home/osmani/src/autosegmentation/"    
    project_path = os.path.join(base_path,"TeethSegmentation/")
    metrics_path = os.path.join(base_path,"Logs/",arch + "/")
    if not os.path.exists(metrics_path):
        os.makedirs(metrics_path)
    
    log_file =  metrics_path + 'losses_metrics_vs_epoch_'
    while os.path.exists(log_file +"%s.csv" % log_index):
        log_index += 1
    log_file =  metrics_path + 'losses_metrics_vs_epoch_' + f"{log_index}.csv"

    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
    
    model_path = os.path.join(project_path, f"models/{arch}/")
    model_name = "Arcad_Mesh_Segementation"
    checkpoint_name = "latest_checkpoint.tar"

    num_classes = 33
    num_channels = 15 #number of features
    num_epochs = 1500
    num_workers = 0
    train_batch_size = 10
    val_batch_size = 10
    num_batches_to_print = 20

    # mkdir 'models'
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    # set dataset
    training_dataset = Mesh_Dataset(from_docker = False, arch = arch, is_train_data = True, train_split = 0.8, patch_size = 6000, model_use=model_use)
    training_dataset.use_mld = False
    val_dataset = Mesh_Dataset(from_docker = False, arch = arch, is_train_data = False, train_split = 0.8, patch_size = 6000, model_use=model_use)
    val_dataset.use_mld = False

    train_loader = DataLoader(dataset=training_dataset,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers)

    val_loader = DataLoader(dataset=val_dataset,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers)

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
    if model_use == 'imeshsegnet:':
        model = iMeshSegNet(num_classes=num_classes, num_channels=num_channels, with_dropout=True, dropout_p=0.5).to(device, dtype=torch.float)
    opt = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)

    losses, mdsc, msen, mppv = [], [], [], []
    val_losses, val_mdsc, val_msen, val_mppv = [], [], [], []
    best_val_dsc = 0.0

    epoch_init = 0
    checkpoint_file = os.path.join(model_path + checkpoint_name)
    if (os.path.exists(checkpoint_file)):
        checkpoint = torch.load(checkpoint_file, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_init = checkpoint['epoch']
        losses = checkpoint['losses']
        mdsc = checkpoint['mdsc']
        msen = checkpoint['msen']
        mppv = checkpoint['mppv']
        val_losses = checkpoint['val_losses']
        val_mdsc = checkpoint['val_mdsc']
        val_msen = checkpoint['val_msen']
        val_mppv = checkpoint['val_mppv']
        best_val_dsc = max(val_mdsc)
        del checkpoint    
    print(f"best val_dsc so far: {best_val_dsc * 100:0.2f}%")
    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    print('Training model...\n\n')
    class_weights = torch.ones(num_classes).to(device, dtype=torch.float)
    training_dataset.total_epoch = num_epochs
    val_dataset.total_epoch = num_epochs
    for epoch in range(epoch_init+1, num_epochs):
        training_dataset.epoch = epoch
        val_dataset.epoch = epoch
        #start_time = time.perf_counter()
        # training
        model.train()
        running_loss = 0.0
        running_mdsc = 0.0
        running_msen = 0.0
        running_mppv = 0.0
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0
        training_dataset.total_batches = len(train_loader)
        for i_batch, batched_sample in enumerate(train_loader):

            # send mini-batch to device
            inputs = batched_sample['cells'].to(device, dtype=torch.float)
            labels = batched_sample['labels'].to(device, dtype=torch.long)

            lookup_term_1, lookup_term_2 = 'A_S', 'A_L'
            if model_use == 'imeshsegnet':
                lookup_term_1, lookup_term_2 = 'knn_6', 'knn_12'

            term_1 = batched_sample[lookup_term_1].to(device, dtype=torch.float)
            term_2 = batched_sample[lookup_term_2].to(device, dtype=torch.float)
            
            one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

            # zero the parameter gradients
            opt.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs, term_1, term_2)
            loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
            dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
            sen = weighting_SEN(outputs, one_hot_labels, class_weights)
            ppv = weighting_PPV(outputs, one_hot_labels, class_weights)
            loss.backward()
            opt.step()

            # print statistics
            running_loss += loss.item()
            running_mdsc += dsc.item()
            running_msen += sen.item()
            running_mppv += ppv.item()
            loss_epoch += loss.item()
            mdsc_epoch += dsc.item()
            msen_epoch += sen.item()
            mppv_epoch += ppv.item()

            training_dataset.running_batch = i_batch + 1
            training_dataset.running_loss = running_loss/(i_batch + 1)
            training_dataset.running_mdsc = running_mdsc/(i_batch + 1)
            training_dataset.running_msen = running_msen/(i_batch+1)
            training_dataset.running_mppv = running_mppv/(i_batch + 1)
            # if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
            #     print('[Epoch: {0}/{1}, Batch: {2}/{3}] loss: {4}, dsc: {5}, sen: {6}, ppv: {7}'.format(epoch+1, num_epochs, i_batch+1, len(train_loader), running_loss/num_batches_to_print, running_mdsc/num_batches_to_print, running_msen/num_batches_to_print, running_mppv/num_batches_to_print))                
            #     running_loss = 0.0
            #     running_mdsc = 0.0
            #     running_msen = 0.0
            #     running_mppv = 0.0

        # record losses and metrics
        losses.append(loss_epoch/len(train_loader))
        mdsc.append(mdsc_epoch/len(train_loader))
        msen.append(msen_epoch/len(train_loader))
        mppv.append(mppv_epoch/len(train_loader))

        #reset
        loss_epoch = 0.0
        mdsc_epoch = 0.0
        msen_epoch = 0.0
        mppv_epoch = 0.0

        # validation
        model.eval()
        with torch.no_grad():
            running_val_loss = 0.0
            running_val_mdsc = 0.0
            running_val_msen = 0.0
            running_val_mppv = 0.0
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0
            training_dataset.total_batches = len(val_loader)
            for i_batch, batched_val_sample in enumerate(val_loader):

                lookup_term_1, lookup_term_2 = 'A_S', 'A_L'
                if model_use == 'imeshsegnet':
                    lookup_term_1, lookup_term_2 = 'knn_6', 'knn_12'

                term_1 = batched_sample[lookup_term_1].to(device, dtype=torch.float)
                term_2 = batched_sample[lookup_term_2].to(device, dtype=torch.float)

                # send mini-batch to device
                inputs = batched_val_sample['cells'].to(device, dtype=torch.float)
                labels = batched_val_sample['labels'].to(device, dtype=torch.long)
                one_hot_labels = nn.functional.one_hot(labels[:, 0, :], num_classes=num_classes)

                outputs = model(inputs, term_1, term_2)
                loss = Generalized_Dice_Loss(outputs, one_hot_labels, class_weights)
                dsc = weighting_DSC(outputs, one_hot_labels, class_weights)
                sen = weighting_SEN(outputs, one_hot_labels, class_weights)
                ppv = weighting_PPV(outputs, one_hot_labels, class_weights)

                running_val_loss += loss.item()
                running_val_mdsc += dsc.item()
                running_val_msen += sen.item()
                running_val_mppv += ppv.item()
                val_loss_epoch += loss.item()
                val_mdsc_epoch += dsc.item()
                val_msen_epoch += sen.item()
                val_mppv_epoch += ppv.item()

                val_dataset.running_batch = i_batch + 1
                val_dataset.running_loss = running_val_loss/(i_batch + 1)
                val_dataset.running_mdsc = running_val_mdsc/(i_batch + 1)
                val_dataset.running_msen = running_val_msen/(i_batch+1)
                val_dataset.running_mppv = running_val_mppv/(i_batch + 1)

                # if i_batch % num_batches_to_print == num_batches_to_print-1:  # print every N mini-batches
                #     print()
                #     print('[Epoch: {0}/{1}, Val batch: {2}/{3}] val_loss: {4}, val_dsc: {5}, val_sen: {6}, val_ppv: {7}'.format(epoch+1, num_epochs, i_batch+1, len(val_loader), running_val_loss/num_batches_to_print, running_val_mdsc/num_batches_to_print, running_val_msen/num_batches_to_print, running_val_mppv/num_batches_to_print))
                #     print('-')
                #     print()
                #     running_val_loss = 0.0
                #     running_val_mdsc = 0.0
                #     running_val_msen = 0.0
                #     running_val_mppv = 0.0

            # record losses and metrics
            val_losses.append(val_loss_epoch/len(val_loader))
            val_mdsc.append(val_mdsc_epoch/len(val_loader))
            val_msen.append(val_msen_epoch/len(val_loader))
            val_mppv.append(val_mppv_epoch/len(val_loader))

            # reset
            val_loss_epoch = 0.0
            val_mdsc_epoch = 0.0
            val_msen_epoch = 0.0
            val_mppv_epoch = 0.0

            # output current status
            print(f'\n*****\nEpoch: {epoch}/{num_epochs}, \n *Training*   : loss: {losses[-1]:0.5f}, dsc: {mdsc[-1]:0.5f}, sen: {msen[-1]:0.5f}, ppv: {mppv[-1]:0.5f}\n *Validating* : loss: {val_losses[-1]:0.5f}, dsc: {val_mdsc[-1]:0.5f}, sen: {val_msen[-1]:0.5f}, ppv: {val_mppv[-1]:0.5f}\n*****\n\n')

        # #one time adjustment. Remove first 25 rows of each column
        # # adjust losses and metrics
        # losses = losses[25:]
        # mdsc = mdsc[25:]
        # msen = msen[25:]
        # mppv = mppv[25:]
        # val_losses = val_losses[25:]
        # val_mdsc = val_mdsc[25:]
        # val_msen = val_msen[25:]
        # val_mppv = val_mppv[25:]
    
        # save the checkpoint
        torch.save({'epoch': epoch+1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'losses': losses,
                    'mdsc': mdsc,
                    'msen': msen,
                    'mppv': mppv,
                    'val_losses': val_losses,
                    'val_mdsc': val_mdsc,
                    'val_msen': val_msen,
                    'val_mppv': val_mppv},
                    model_path+checkpoint_name)

        # save the best model
        if best_val_dsc < val_mdsc[-1]:
            best_val_dsc = val_mdsc[-1]
            torch.save({'epoch': epoch+1,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                        'losses': losses,
                        'mdsc': mdsc,
                        'msen': msen,
                        'mppv': mppv,
                        'val_losses': val_losses,
                        'val_mdsc': val_mdsc,
                        'val_msen': val_msen,
                        'val_mppv': val_mppv},
                        model_path+'{}_best.tar'.format(model_name))

        # save all losses and metrics data
        pd_dict = {'loss': losses, 'DSC': mdsc, 'SEN': msen, 'PPV': mppv, 'val_loss': val_losses, 'val_DSC': val_mdsc, 'val_SEN': val_msen, 'val_PPV': val_mppv}
        stat = pd.DataFrame(pd_dict)
        stat.to_csv( log_file)
        #elapsed = time.perf_counter()

