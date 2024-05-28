import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from torch.autograd import Variable
from DenseGait_BaseOpenGait.model.modules import *
from DenseGait_BaseOpenGait.model.modules import _P3DA,_P3DB,_P3DC,CBAM,CAM,SAM

class DenseLayer(nn.Module):
    def __init__(self,block_type, num_input_features, growth_rate, bn_size, drop_rate):  # ,block_dilation,block_padding
        super(DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm3d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(in_channels=num_input_features, out_channels=bn_size * growth_rate,
                                           kernel_size=1, stride=1, bias=False)

        self.norm2 = nn.BatchNorm3d(bn_size * growth_rate)


        if block_type == 0:
            self.conv2 = _P3DA(num_input_features=bn_size * growth_rate, growth_rate=growth_rate)
        elif block_type == 1:
            self.conv2 = _P3DB(num_input_features=bn_size * growth_rate, growth_rate=growth_rate)
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.norm1(x)
        new_features = self.relu(new_features)
        new_features = self.conv1(new_features)
        new_features = self.norm2(new_features)
        new_features = self.relu(new_features)
        new_features = self.conv2(new_features)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate)
        out = torch.cat([x, new_features], 1)
        return out

def _make_layer(block,block_type,num_layers,num_input_features,bn_size,growth_rate,drop_rate):
    layers = []
    for i in range(num_layers):
        layers.append(block(block_type, num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate))#,t,h,w
    return nn.Sequential(*layers)


class DenseGait(nn.Module):
    def __init__(self,block, block_type=(), growth_rate=(), block_config = (),num_init_features=64,bn_size=4,
                 compression_rate=0.5,drop_rate=0.5):

        super(DenseGait, self).__init__()
        self.drop_rate = drop_rate
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=num_init_features, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = nn.BatchNorm3d(num_init_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = T_Conv(in_planes=num_init_features, out_planes=num_init_features,kernal_size=3, stride=(3,1,1), padding = (1,0,0))

        self.pooling = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        num_features = num_init_features
        self.block1 = _make_layer(block,block_type=block_type[0],num_layers=block_config[0],num_input_features=num_features,
                                  bn_size=bn_size,growth_rate=growth_rate[0],drop_rate=drop_rate)
        num_features = num_features + int(block_config[0]) * growth_rate[0]
        self.CBAM1 = CBAM(channel=num_features)


        num_features = num_features * compression_rate

        self.block2 = _make_layer(block,block_type=block_type[1],num_layers=block_config[1],num_input_features=num_features,
                                  bn_size=bn_size,growth_rate=growth_rate[1],drop_rate=drop_rate)
        num_features = num_features + int(block_config[1]) * growth_rate[1]
        self.CBAM2 = CBAM(channel=num_features)
  

        self.block3 = _make_layer(block,block_type=block_type[2],num_layers=block_config[2],num_input_features=num_features,
                                  bn_size=bn_size,growth_rate=growth_rate[2],drop_rate=drop_rate)
        self.CBAM3 = CBAM(channel=num_features + block_config[2] * growth_rate[2])
        num_features = num_features + int(block_config[2]) * growth_rate[2]
    
        self.normfinal = nn.BatchNorm3d(num_features)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.conv2(x)

        x = self.block1(x)
        x = self.CBAM1(x)

        x = self.pooling(x)

        x = self.block2(x)
        x = self.CBAM2(x)

        x = self.block3(x)
        x = self.CBAM3(x)
    
        x = self.normfinal(x)
        x = self.relu(x)
        return x



def DenseGait3D():
    return DenseGait(block=DenseLayer, block_type=(0,0,0), growth_rate=(32, 64, 128), block_config=(1, 1, 1), num_init_features=32, bn_size=2, compression_rate=1, drop_rate=0.5)

def DenseGait3D_OUMVLP():

    return DenseGait(block=DenseLayer, block_type=(1,1,0,1), growth_rate=(32, 64, 128, 256), block_config=(1, 1, 1, 1), num_init_features=32, bn_size=2,
                     compression_rate=1, drop_rate=0.5)

