import numpy as np
import torch.nn as nn
import h5py
from torch.utils.data import Dataset


# source: https://github.com/yjn870/FSRCNN-pytorch/blob/master/datasets.py
class TrainDataset(Dataset):
    def __init__(self, h5_file):
        super(TrainDataset, self).__init__()
        self.h5_file = h5_file

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            return np.expand_dims(f['lr'][idx], 0), np.expand_dims(f['hr'][idx], 0)

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['lr'])


# reference: https://towardsdatascience.com/review-fsrcnn-super-resolution-80ca2ee14da4
class fsrcnn(nn.Module):
    def __init__(self,d,s,m,x):
        super(fsrcnn, self).__init__()
        self.x = x

        # Conv(f,n,c): f = fxf filter size, n = #filters, c = #input channels

        # feature extraction: Conv(5,d,1)
        self.feat_extrac = nn.Sequential(
            nn.Conv2d(in_channels=1,out_channels=d,kernel_size=5),
            nn.PReLU(d)
        )
    
        # shrinking: Conv(1,s,d)
        self.shrink = nn.Sequential(
            nn.Conv2d(in_channels=d,out_channels=s,kernel_size=1),
            nn.PReLU(s)
        )

        # mapping for m times: Conv(3,s,s)
        self.mapping = nn.Sequential()
        for i in range(m):
            self.mapping.add_module('map_{}'.format(i), nn.Conv2d(in_channels=s,
                                                            out_channels=s,
                                                            kernel_size=3))
            self.mapping.add_module('map_prelu_{}'.format(i),nn.PReLU(s))

    
        # expanding: Conv(1,d,s)
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels=s,out_channels=d,kernel_size=1),
            nn.PReLU(d)
        )

        self.DeConvolution(d)

        # TODO: initial weights?

    def DeConvolution(self,d):
        if self.x == 2:
            # deconvolution: DeConv(9,1,s)
            # out_channels = number of the image channel, 1 for this project
            # # m = 2
            # self.deconv = nn.ConvTranspose2d(in_channels=d,out_channels=1,
            #                                  kernel_size=9,stride=self.x,dilation=2,output_padding=1)
            # m = 3
            self.deconv = nn.ConvTranspose2d(in_channels=d,out_channels=1,padding=2,
                                            kernel_size=9,stride=self.x,dilation=3,output_padding=1)
            # # m = 4
            # self.deconv = nn.ConvTranspose2d(in_channels=d,out_channels=1,
            #                                 kernel_size=9,stride=self.x,dilation=3,output_padding=1)
        elif self.x == 3:
            # m = 3
            self.deconv = nn.ConvTranspose2d(in_channels=d,out_channels=1,
                                            kernel_size=9,stride=self.x,dilation=4)
            # # m = 4
            # self.deconv = nn.ConvTranspose2d(in_channels=d,out_channels=1,padding=1,
            #                                 kernel_size=9,stride=self.x,dilation=5)
        else:
            # # m = 4 
            # self.deconv = nn.ConvTranspose2d(in_channels=d,out_channels=1,output_padding=3,
            #                                  kernel_size=9,stride=self.x,dilation=6)
            # m = 1
            self.deconv = nn.ConvTranspose2d(in_channels=d,out_channels=1,output_padding=3,
                                            kernel_size=9,stride=self.x,dilation=3)


    def forward(self,x):
        x = self.shrink(self.feat_extrac(x))
        x = self.mapping(x)
        x = self.expand(x)
        x = self.deconv(x)
        return x