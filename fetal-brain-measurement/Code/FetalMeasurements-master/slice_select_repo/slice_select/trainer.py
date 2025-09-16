
from __future__ import print_function

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.autograd import Variable
from tensorboardX import SummaryWriter
from torch.optim import lr_scheduler
import time
import copy
import os



from loader import NiftiDataset, random_split
import transforms as tfs

from sacred import Experiment
from sacred.commands import print_config
from sacred.observers import MongoObserver

save_dir = os.path.join(".", "models")

writer = SummaryWriter('runs')
writer.add_text('mini', 'minimal model', 0)


ex = Experiment('slicenet')
# ex.observers.append(MongoObserver.create(url=f'mongodb://sample:password@localhost:27017/?authMechanism=SCRAM-SHA-1', db_name='db'))

def getTimeName():
    """Return the current time in format <day>-<month>_<hour><minute> for use in filenames."""
    from datetime import datetime
    t = datetime.now()
    return "{:02d}-{:02d}_{:02d}{:02d}".format(t.day,t.month,t.hour,t.minute)

def save_model(model, epoch, directory, metrics, filename=None):
    """Save the state dict of the model in the directory,
    with the save name metrics at the given epoch.
    epoch: epoch number(<= 4 digits)
    directory: where to save statedict
    metrics: dictionary of metrics to append to save filename
    filename: if a name is given, it overrides the above created filename
    Returns the save file name
    """
    # save state dict
    postfix = ""
    if filename is None:
        filename = f"epoch{epoch:04d}_{getTimeName()}_"
        postfix = "_".join([f"{name}{val:0.4f}" for name, val in metrics.items()])

    filename = os.path.join(directory, filename + postfix + ".statedict.pkl")

    if isinstance(model, nn.DataParallel):
        state = model.module.state_dict()
    else:
        state = model.state_dict()

    torch.save(state, filename)
    print(f"Saved model at {filename}")

    return filename


@ex.capture
def create_dataloaders(cuda, batch_size, db_params, data):
    #Input validation
    if not os.path.exists(db_params['csv']):
        raise ValueError('CSV file not exists {}'.format(db_params['csv']))

    if not os.path.isdir(db_params['root_dir']):
        raise ValueError('Root dir for DB does not exists {}'.format(db_params['root_dir']))
    #Dataloader creation


    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda in [0,1] else {}
    dataset = NiftiDataset(csv_file=db_params['csv'],
                           nii_dir=db_params['root_dir'],
                           seg_dir=db_params['seg_dir'],
                           filter_quality=db_params['quality'],
                           only_tag=True,
                           tagname=data["selection_idx"],
                           transform=transforms.Compose([

                            tfs.RandomRotate(),
                            tfs.cropByBBox(min_upcrop=1.0, max_upcrop=1.3),
                            tfs.PadZ(data['context']),
                            tfs.Rescale((224,224)),

                            tfs.ToTensor(),
                            tfs.Normalize(mean=0.456,
                                            std=0.224),
                            tfs.SampleFrom3D(db_params['pos_neg_ratio'],
                                             sample_idx=data['selection_idx'],
                                             context=data['context']),
                            #tfs.RandomFlip(),
                            tfs.toXY("image", data['selection_idx']),
                           ]))
    dataset_size = len(dataset) 
    train_size = int(db_params['train_test_split'] * dataset_size)
    test_size = dataset_size - train_size
    (train_ds, test_ds) , (train_idx, test_idx) = random_split(dataset, [train_size, test_size])

    #It is not going to be elegant.... special augmentation for test
    test_ds.dataset = copy.copy(dataset)
    test_ds.dataset.transform = transforms.Compose([ 

                                            tfs.cropByBBox(min_upcrop=None, max_upcrop=None),
                                            tfs.PadZ(data['context']),
                                            tfs.Rescale((224,224)),

                                            tfs.ToTensor(),
                                            tfs.Normalize(mean=0.456,
                                                            std=0.224),
                                            tfs.SampleFrom3D(None,
                                                             sample_idx=data['selection_idx'],
                                                             context=data['context']),
                                            #tfs.RandomFlip(),
                                            tfs.toXY("image", data['selection_idx']),
                                           ])
    train_val_ds = copy.copy(test_ds)
    train_val_ds.indices = train_ds.indices

    ex.info["db_split_train"] = dataset.get_metadata(train_idx).to_json()
    ex.info["db_split_test"] = dataset.get_metadata(test_idx).to_json()
    train_loader = torch.utils.data.DataLoader(train_ds,
                                    batch_size=batch_size, shuffle=True, collate_fn=tfs.custom_collate_fn, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_ds,
                                    batch_size=1, shuffle=True, collate_fn=tfs.custom_collate_fn, **kwargs)
    train_val_loader = torch.utils.data.DataLoader(train_val_ds,
                                    batch_size=1, shuffle=True, collate_fn=tfs.custom_collate_fn, **kwargs)


    dataloaders = {}
    dataloaders["train"] = train_loader
    dataloaders["val"] = test_loader
    dataloaders["val_train"] = train_val_loader
    return dataloaders

@ex.capture
def create_model(optimizer_params):
    #model_ft = models.resnet18(pretrained=True)
    model_ft = models.resnet34(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=optimizer_params['lr'], momentum=optimizer_params['momentum'])

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=optimizer_params['step_size'], gamma=optimizer_params['gamma'])
    #exp_lr_scheduler= lr_scheduler.CyclicLR(optimizer_ft, base_lr=optimizer_params['lr'], max_lr=optimizer_params['lr']*10)

    return model_ft, criterion, optimizer_ft, exp_lr_scheduler

@ex.capture
def train_model(model, criterion, optimizer, scheduler, dataloaders, device,_run, num_epochs, optimizer_params):
    since = time.time()
    run_dir = os.path.join(save_dir, str(_run._id))
    os.makedirs(run_dir, exist_ok=True)
    model = model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_model_epoch = 0

    

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        if epoch == optimizer_params['freeze_from']:
            print('Freezing model')
            for param in model.parameters():
                param.requires_grad = False
            #Unfreeze last layer
            for param in model.fc.parameters():
                param.requires_grad = True
            for param in model.conv1.parameters():
                param.requires_grad = True

        #if epoch % 10 == 0:
        #dataloaders['train'].dataset.dataset.transform.transforms[6].negative_slides = int((30 - epoch) / 3) + 1
        #dataloaders['train'].dataset.dataset.transform.transforms[6].negative_slides = int(((epoch) / 3)) + 1

        # Each epoch has a training and validation phase
        for phase in ['train', 'val', 'val_train']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0
            running_choose_acc = 0.0
            running_shift = 0.0
            
            epoch_elem_size = 0
            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                epoch_elem_size += inputs.size(0)
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                if phase in ['val', 'val_train']:
                    output_sfmax = torch.nn.functional.softmax(outputs,dim=1)
                    max_val, max_idx = torch.max(output_sfmax[:,1],dim=0)
                    max_val_lbl, max_val_idx = torch.max(labels.data,dim=0)
                    running_choose_acc += (1. - (abs(float(max_idx) - float(max_val_idx))/float(inputs.size(0))))
                    running_shift += abs(float(max_idx) - float(max_val_idx))
                    #print(max_idx, max_val_idx, running_idx)
                    

            epoch_loss = running_loss / epoch_elem_size
            epoch_acc = running_corrects.double() / epoch_elem_size
            if phase in ['val', 'val_train']:
                epoch_choose_acc = running_choose_acc / len(dataloaders[phase])
                epoch_choose_shift = running_shift / len(dataloaders[phase])

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            if phase == 'val':

                #Calculate specific Val accuracy
                #Assuming BS=1 in Val
                _run.log_scalar('val_choose_acc', float(epoch_choose_acc))
                _run.log_scalar('val_choose_shift', float(epoch_choose_shift))
                _run.log_scalar('val_loss', float(epoch_loss))
                _run.log_scalar('val_accuracy', float(epoch_acc))
            elif phase == 'val_train':
                _run.log_scalar('vtrain_choose_acc', float(epoch_choose_acc))
                _run.log_scalar('vtrain_choose_shift', float(epoch_choose_shift))
                _run.log_scalar('vtrain_loss', float(epoch_loss))
                _run.log_scalar('vtrain_accuracy', float(epoch_acc))
            elif phase == 'train':
                _run.log_scalar('train_loss', float(epoch_loss))
                _run.log_scalar('train_accuracy', float(epoch_acc))


            # deep copy the model
            if phase == 'val' and epoch_choose_acc > best_acc:
                best_acc = epoch_choose_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                best_model_epoch = epoch

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # Save last epoch model
    model_fname = save_model(model, epoch, run_dir, {'choose_acc' : epoch_choose_acc})
    _run.add_artifact(model_fname, "model_lastepoch")

    # load best model weights
    model.load_state_dict(best_model_wts)

    model_fname = save_model(model, best_model_epoch, run_dir, {'choose_acc' : best_acc})
    _run.add_artifact(model_fname, "model")
    return model

@ex.config
def get_config():
    batch_size = 6
    num_epochs = 25
    cuda = 1
    data = {
        'context': 1,
        'selection_idx' : 'TCD_Selection',
    }
    db_params = {
        'root_dir' : '/media/df3-dafna/Netanell/BrainCorToGal_Fixed/',
        'seg_dir' : '/media/df3-dafna/Netanell/BLLAMODEL/',
        'csv' : '/media/df3-dafna/Netanell/BLLAMODEL/Data.xlsx',
        'quality' : 3,
        'pos_neg_ratio' : 2,
        'train_test_split' : 0.8,
    }
    optimizer_params = {
        'lr' : 0.001,
        'momentum' : 0.9,
        'step_size' : 7,
        'gamma' : 0.1,
        'freeze_from' : -1,
    }


# @ex.config
# def initialize(cuda):#, seed):
    # # Training settings
    # parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    # parser.add_argument('--batch-size', type=int, default=8, metavar='N',
    #                     help='input batch size for training (default: 64)')
    # parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
    #                     help='input batch size for testing (default: 1000)')
    # parser.add_argument('--epochs', type=int, default=2, metavar='N',
    #                     help='number of epochs to train (default: 10)')
    # parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
    #                     help='learning rate (default: 0.01)')
    # parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
    #                     help='SGD momentum (default: 0.5)')
    # parser.add_argument('--no-cuda', action='store_true', default=False,
    #                     help='disables CUDA training')
    # parser.add_argument('--seed', type=int, default=1, metavar='S',
    #                     help='random seed (default: 1)')
    # parser.add_argument('--log-interval', type=int, default=10, metavar='N',
    #                     help='how many batches to wait before logging training status')
    # args = parser.parse_args(args_str)
    # args.cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(seed)
    # if cuda > 0:
    #     torch.cuda.manual_seed(seed)

@ex.capture()
def create_device(cuda):
    cuda_int = int(cuda)
    torch.cuda.set_device(cuda_int)
    device = torch.device("cuda:%d" % (cuda_int, ))
    return device

@ex.main
def main():
    # initialize()
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = create_device()
    dataloaders = create_dataloaders()

    model, criterion, optimizer, scheduler = create_model()
    model = train_model(model, criterion, optimizer, scheduler, dataloaders, device)
    

if __name__ == '__main__':
    ex.run_commandline()
