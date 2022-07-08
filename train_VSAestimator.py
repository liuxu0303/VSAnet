import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import argparse
import time
import datetime
from path import Path
from tqdm import tqdm
import numpy as np
from loss import  AreaBinsChamferLoss
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from utils import  save_checkpoint
from sequence_folders import SequenceFolder
from torchvision import transforms
from models.VSAestimator import VSAestimator



parser = argparse.ArgumentParser(description='VSA estimation network training on VSA estimation Dataset ',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data', metavar='DIR', help='path to dataset', default='./path to VSA estimation dataset')
parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence', help='the dataset dype to train')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=48, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.00015, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--base-momentum', default=0.85, type=float, metavar='M', help='base_momentum for OneCycleLR')
parser.add_argument('--max-momentum', default=0.95, type=float, metavar='M', help='max_momentum for OneCycleLR')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0.1, type=float, metavar='W', help='weight decay')
parser.add_argument('--div-factor', default=25, type=int, help='div_factor')
parser.add_argument('--final-div-factor', default=100, type=int, help='final_div_factor')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--dataset', type=str, choices=['VSA300'], default='VSA300', help='the dataset to train')
parser.add_argument('--pretrained-vsa', dest='pretrained_vsa', default=None, metavar='PATH', help='path to pre-trained VSA estimation model')
parser.add_argument('--name', dest='name', type=str, default='new', help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument("--local-rank",default = None, type=int)
parser.add_argument("--num-classes",default = 9, type=int, help="the number of categories")
parser.add_argument("--n-bins",default = 256, type=int, help="area bins numbers")
parser.add_argument("--num-layers",default = 3, type=int, help="transformer layer numbers")
parser.add_argument("--distributed", default=True, action="store_true", help="Use DDP if set")
parser.add_argument("--alpha",default = 2.5, type=float, help="the weights of VSA training loss ")
parser.add_argument("--beta",default = 0.1, type=float, help="the weights of A_max training loss ")
parser.add_argument("--gama",default = 0.1, type=float, help="the weights of area_bin_centers training loss ")


best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


torch.autograd.set_detect_anomaly(True)


def main():
    
    global best_error, n_iter, device
    args = parser.parse_args()
    a = time.time()
    timestamp = datetime.datetime.now().strftime("%m-%d-%H-%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

 
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                            std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose([  
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.3),
        transforms.ToTensor(),
        normalize,
    ])

#    valid_transform = transforms.Compose([transforms.ToTensor(), normalize,])

    print("=> fetching scenes in '{}'".format(args.data))
    if args.folder_type == 'sequence':
        train_set = SequenceFolder(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            dataset=args.dataset
        )
       
    print('{} samples found'.format(len(train_set.samples)))

    # create model
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(backend='nccl', init_method='tcp://localhost: 23456', rank=0, world_size=1)
    print("=> creating model")
    #VSA estimation network with efficientnet_b5
    model = VSAestimator(pretrained=True, num_classes=args.num_classes, n_bins=args.n_bins, num_layers=args.num_layers).cuda()
    
#    summary(model, input_size=[(3, 300, 300)], batch_size=48, device='cpu')
    
    if args.pretrained_vsa:
        print("=> using pre-trained weights for VSANet")
        weights = torch.load(args.pretrained_vsa)
        model.load_state_dict(weights['state_dict'], strict=False)
        print("Done！")

     
#    model = torch.nn.DataParallel(model)
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)
  
    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)
     
    print('=> setting adam solver')    
    m = model.module 
    optim_params = [{"params": m.get_1x_lr_params(), "lr": args.lr / 10},
                   {"params": m.get_10x_lr_params(), "lr": args.lr}]
    optimizer = torch.optim.AdamW(optim_params, weight_decay=args.weight_decay, lr=args.lr)
    epochs = args.epochs
    args.last_epoch = -1
    args.epoch = 0
    
    ###################################### Scheduler ###############################################
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, args.lr, epochs=epochs, steps_per_epoch=len(train_loader),
                                              cycle_momentum=True,
                                              base_momentum=args.base_momentum, max_momentum=args.max_momentum, last_epoch=args.last_epoch,
                                              div_factor=args.div_factor,
                                              final_div_factor=args.final_div_factor)
    
    if args.pretrained_vsa and scheduler is not None:
        scheduler.step(args.epoch + 1)

    best_loss = np.inf
    
    for epoch in range(args.epoch, args.epochs):
        
        if args.distributed:
            train_sampler.set_epoch(epoch) 
            
        train_loss, arealoss_sum, classloss_sum, areamaxloss_sum, areabinloss_sum = train(args, train_loader, model, optimizer, scheduler, args.epoch_size)
      
        arealoss = arealoss_sum * args.batch_size/len(train_set)
        
        if epoch % 1 == 0:
            save_checkpoint(
                   args.save_path,
                   timestamp,
                   {
                       'epoch': epoch + 1,
                       'state_dict': model.module.state_dict()
                   }, 
                   filename='last_checkpoint.pth.tar'
            )  
            
            if arealoss < best_loss:
                save_checkpoint(
                       args.save_path,
                       timestamp,
                       {
                           'epoch': epoch + 1,
                           'state_dict': model.module.state_dict()
                       }, 
                       filename='best_checkpoint.pth.tar'
                )  
                best_loss = arealoss
       
        b = (time.time()-a)/60
        print('epoch: ', epoch, ' of ', args.epochs, 'loss:', train_loss * args.batch_size/len(train_set), \
              'arealoss:', arealoss_sum * args.batch_size/len(train_set), 'classloss:', classloss_sum * args.batch_size/len(train_set),\
              'areamaxloss:', areamaxloss_sum * args.batch_size/len(train_set),'areabinloss:', areabinloss_sum * args.batch_size/len(train_set),b, 'min\n')
        a = time.time()

def train(args, train_loader, model, optimizer, scheduler, epoch_size):
    global n_iter, device
    
    SmoothL1Loss = nn.SmoothL1Loss().cuda()
    crossentropyloss = nn.CrossEntropyLoss().cuda()
    criterion_areabins = AreaBinsChamferLoss().cuda()
    
    model.train()

    allloss = 0
    arealoss_sum = 0
    classloss_sum = 0
    areamaxloss_sum = 0
    areabinloss_sum = 0

    for i, (img, arealabel, classlabel, set_V, areamax, _) in enumerate(tqdm(train_loader)):

        img = img.cuda()
        arealabel = arealabel.cuda().reshape(arealabel.size()[0],1)
        classlabel = classlabel.cuda()
        set_V = set_V.cuda().reshape(arealabel.size()[0],-1)
        areamax = areamax.cuda().reshape(areamax.size()[0],1)

        # compute output
        optimizer.zero_grad()
        
        areabin_edges, vsa_pred, cate_dist, areamax_pred = model(img)
#         
        arealoss = SmoothL1Loss( vsa_pred ,  arealabel ) # the training loss of the prediction of VSA
        areamaxloss = SmoothL1Loss( areamax_pred ,  areamax ) # the training loss of the prediction of A_max        
        classloss = crossentropyloss(cate_dist, classlabel) # the loss for the classification module
        areabinloss = criterion_areabins(areabin_edges, set_V) # the training loss of the prediction of the area-bin-centers
      
               
        loss  = args.alpha * arealoss + classloss + args.beta * areamaxloss + args.gama * areabinloss

        #optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.1)  # optional
        optimizer.step()
        
        scheduler.step()
        
        # del b
        arealoss = arealoss.sum()
        classloss = classloss.sum()
        areamaxloss = areamaxloss.sum()
        areabinloss = areabinloss.sum()
        allloss += float(loss.abs())
        arealoss_sum += float(arealoss.abs())
        classloss_sum += float(classloss.abs())
        areamaxloss_sum += float(areamaxloss.abs())
        areabinloss_sum += float(areabinloss.abs())
       
        del loss, arealoss, classloss, areabinloss, areamaxloss

    return allloss, arealoss_sum, classloss_sum, areamaxloss_sum, areabinloss_sum
 

if __name__ == '__main__':
    main()
