import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from FSRCNN_model import TrainDataset, fsrcnn


params_x2 = {'lr':0.001,'epochs':50,'d':40,'s':15,'m':3,'batch':30}
params_x3 = {'lr':0.001,'epochs':50,'d':25,'s':20,'m':3,'batch':30}
params_x4 = {'lr':0.001,'epochs':50,'d':40,'s':15,'m':1,'batch':30}


def PSNR(mse):
    if(mse == 0):
        return 100
    return 10*np.log10((1**2)/mse)


def train_model(datafile,x,params):
    # components for the training process
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = fsrcnn(d=params['d'],s=params['s'],m=params['m'],x=x).to(device)
    criterion = nn.MSELoss().to(device)
    loader_train = DataLoader(dataset=TrainDataset(datafile),batch_size=params['batch'],shuffle=True)
    optimizer = optim.Adam([{'params':model.feat_extrac.parameters()},
                            {'params':model.shrink.parameters()},
                            {'params':model.mapping.parameters()},
                            {'params':model.expand.parameters()},
                            {'params':model.deconv.parameters()}
                            ], lr=params['lr'])
  
    # run epochs
    mse_loss, psnr = [], []
    for epoch in np.arange(0,params['epochs']):
        model.train()

        for row in loader_train:
            # read data and move it to GPU
            lr, hr = row
            lr, hr = lr.to(device,non_blocking=True), hr.to(device,non_blocking=True)

            # forward pass
            outputs = model(lr)
            loss = criterion(outputs,hr)

            # backprop
            optimizer.zero_grad()
            loss.backward()

            # optimize
            optimizer.step()
        mse_loss  += [loss.item()]
        psnr += [PSNR(loss.item())]

    return mse_loss, psnr, model


def test_model(model,datafile,params):
    loader_valid = DataLoader(dataset=TrainDataset(datafile),batch_size=params['batch'])
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mse = nn.MSELoss().to(device)
    mse_loss,psnr = [],[]

    with torch.no_grad():
        for lr, hr in loader_valid:
            lr, hr = lr.to(device,non_blocking=True), hr.to(device,non_blocking=True)
            model.eval()
            outputs = model(lr)

            # metrics
            loss = mse(outputs,hr)
            mse_loss  += [loss.item()]
            psnr += [PSNR(loss.item())]

    return mse_loss, psnr


# example usage:
# mse_train_x2, psnr_train_x2, model_x2 = train_model('dataset/train_x2.h5py',2,params_x2)
# mse_valid_x2, psnr_valid_x2 = test_model(model_x2,'dataset/valid_x2.h5py',params_x2)

# mse_train_x3, psnr_train_x3, model_x3 = train_model('dataset/train_x3.h5py',3,params_x3)
# mse_valid_x3, psnr_valid_x3 = test_model(model_x3,'dataset/valid_x3.h5py',params_x3)

# mse_train_x4, psnr_train_x4, model_x4 = train_model('dataset/train_x4.h5py',4,params_x4)
# mse_valid_x4, psnr_valid_x4 = test_model(model_x4,'dataset/valid_x4.h5py',params_x4)