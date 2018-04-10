import os
import numpy as np
import h5py
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
from decimal import Decimal
import torch 
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
from glob import glob
import torch
from torch.autograd import Variable
import time
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data.distributed
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
from utils import progress_bar

class EventDataset(Dataset):
    def __init__(self, jets, constituents, targets,
                 constituents_name = ['constituents_pt', 'constituents_eta', 'constituents_phi', 
                                      'constituents_charge', 'constituents_dxy', 'constituents_dz', 
                                      'constituents_Eem', 'constituents_Ehad'], 
                 jets_name = ['recojet_pt', 'recojet_eta', 'recojet_phi', 'recojet_m', 'recojet_sd_pt', 
                             'recojet_sd_eta', 'recojet_sd_phi', 'recojet_sd_m', 'n_constituents']
                ):
        self.jets = jets
        self.constituents = constituents
        self.targets = targets
        self.constituents_name = constituents_name
        self.jets_name = jets_name
    def __len__(self):
        return self.jets.shape[0]
    def __getitem__(self,idx):
        out = (torch.FloatTensor(self.jets[idx,...]),
        torch.FloatTensor(self.constituents[idx,...]),
        torch.FloatTensor(self.targets[idx,...]))
        return out

class SimpleNet(nn.Module):
    def __init__(self, debug=False):
        super(SimpleNet, self).__init__()
        
        self.gru = nn.GRU(input_size=8, hidden_size=200, num_layers=2, batch_first=True, dropout=0.2)
        self.linear1 = nn.Linear(200,50)
        self.linear2 = nn.Linear(9,50)
        self.linear3 = nn.Linear(100,1)
        self.debug = debug
    def forward(self, constituents, jets):
        if self.debug:
            print("constituents = {}".format(constituents.shape))
            print("jets = {}".format(jets.shape))
        self.gru.flatten_parameters()
        _, con = self.gru(constituents)
        if self.debug:
            print("con = {}".format(con.shape))
        con = con[1]
        if self.debug:
            print("con = {}".format(con.shape))
        con = self.linear1(con)
        if self.debug:
            print("lineared con = {}".format(con.shape))
        je = self.linear2(jets)
        if self.debug:
            print("je = {}".format(je.shape))
        merge = torch.cat((con,je), 1)
        if self.debug:
            print ("merge = {}".format(merge.shape))
        out = self.linear3(merge)
        return out

class ResolutionLoss(torch.nn.Module):
    
    def __init__(self):
        super(ResolutionLoss,self).__init__()
        
    def forward(self,x,y,weight):
        epsilon=1e-7
        resolution = weight*((y-x)/(y+epsilon))**2
        return torch.sum(resolution)

def get_weight(pt):
    min_pt = -0.45261003
    max_pt = 1.03705417
    weight = torch.FloatTensor(pt.shape).fill_(10.)
    weight[pt < min_pt] = 1.
    weight[pt > max_pt] = 1.
    return weight
        


def train(epoch):
    global total_train_loss
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(train_loader):
        jets, cons, targets = data[0], data[1], data[2]
        #weights = get_weight(jets[:,0]).cuda()
        optimizer.zero_grad()
        jets, cons, targets = Variable(jets.cuda()), Variable(cons.cuda()), Variable(targets.cuda())
        #weights = Variable(weights)
        outputs = net(cons, jets)
        loss = criterion(outputs, targets) ## more weight to (5000, 7000)
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        total += targets.size(0)
        
        progress_bar(batch_idx, len(train_loader), 'Loss: {:.4E}'.format(Decimal(train_loss/(batch_idx+1))))

    total_train_loss.append(train_loss/(batch_idx+1))
    
def validate(epoch):
    global best_loss
    global total_val_loss
    global total_val_acc
    global total_train_loss
    global total_train_acc
    net.eval()
    test_loss = 0
    total = 0
    for batch_idx, data in enumerate(valid_loader):
        jets, cons, targets = data[0], data[1], data[2]
        #weights = get_weight(jets[:,0]).cuda()
        jets, cons, targets = Variable(jets.cuda(), volatile=True), Variable(cons.cuda(), volatile=True), Variable(targets.cuda())
        #weights = Variable(weights, volatile=True)
        
        outputs = net(cons, jets)
        loss = criterion(outputs, targets)#, weights)

        test_loss += loss.data[0]
        total += targets.size(0)
        current_loss = test_loss/(batch_idx+1)
        
        progress_bar(batch_idx, len(valid_loader), 'Loss: {:.4E}'.format(Decimal(current_loss)))
    
    total_val_loss.append(current_loss)
    # Save checkpoint.
    print("Current loss = {:.4E}, best loss = {:.4E}".format(Decimal(current_loss), Decimal(best_loss)))
    if current_loss < best_loss:
        print('Saving..')
        state = {
            'net': net, #.module if use_cuda else net,
            'epoch': epoch,
            'best_loss':current_loss,
            'train_loss':total_train_loss,
            'val_loss':total_val_loss,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckptMSE.t7')
        best_loss = current_loss
    if epoch == 0: best_loss = current_loss

if __name__ == '__main__':
    import os
    import argparse
    from utils import progress_bar
    import torch.distributed as dist

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('--lr', default=1.e-3, type=float, help='learning rate')
    parser.add_argument('--resume', action='store_true', help='resume from checkpoint')
    parser.add_argument('--dry', action='store_true', help='dry run')
    parser.add_argument('--world-size', default=1, type=int,
                                help='number of distributed processes')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of workers')
    parser.add_argument('-b', '--batch-size', default=1000, type=int,
                    metavar='N', help='mini-batch size')
    parser.add_argument('--rank','-r',default=0, type=int, help = 'Rank of the process')
    parser.add_argument('-d','--device',default='6', help = 'GPUs to use')
    args = parser.parse_args()
    nGPUs = len(args.device.split(','))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    use_cuda = torch.cuda.is_available()

    args.distributed = args.world_size > 1
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    
    valid_size = 0.3
    batch_size = args.batch_size
    pin_memory = False
    num_workers = args.workers

    best_loss = 0
    start_epoch = 0
    total_train_acc = []
    total_val_acc = []
    total_train_loss = []
    total_val_loss = []

    ### Loading dataset
    print("Loading dataset")
    with h5py.File("/bigdata/shared/IML/preprocessed_qcd.h5","r") as infile:
        if args.dry:
            sorted_pt_constituents = infile['Constituents'][:4000]
            scaled_jets = infile['Jets'][:4000]
            mass_targets = infile['Targets'][:4000]
        else:
            sorted_pt_constituents = infile['Constituents'][:]
            scaled_jets = infile['Jets'][:]
            mass_targets = infile['Targets'][:]
    print("Loading completed")

    events = EventDataset(scaled_jets, sorted_pt_constituents, np.expand_dims(mass_targets,axis=1))
    num_train = len(events)
    indices = list(range(len(events)))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = DataLoader(
            events, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )
    valid_loader = DataLoader(
            events, batch_size=batch_size, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=pin_memory,
        )

    if args.resume:
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckptMSE.t7')
        net = checkpoint['net']
        best_loss = checkpoint['best_loss']
        start_epoch = checkpoint['epoch']
        total_train_loss = checkpoint['train_loss']
        total_val_loss = checkpoint['val_loss']
    
    else:   
        print("Creating new networks")
        net = SimpleNet()
    
    if not args.distributed:
        if nGPUs > 1:
            print("Parallelize data on GPUs {}".format(args.device))
            net = torch.nn.DataParallel(net).cuda()
        else:
            print ("Sent net to GPU {}".format(args.device))
            net = net.cuda()
    else:
        net = net.cuda()
        print ("Sent net to GPU")
        net = torch.nn.parallel.DistributedDataParallel(net)
        print ("Distributed net")

    cudnn.benchmark=True

    criterion = nn.MSELoss().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr = 1e-3)
    starttime = time.time()
    for epoch in range(start_epoch, start_epoch+100):
        train(epoch)
        validate(epoch)
    endtime = time.time()
    print("Wall time = {}s".format(endtime-starttime))


        

