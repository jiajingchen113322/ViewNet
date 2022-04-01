import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.pcview import PCViews



class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, bias=False, **kwargs)

    def forward(self, x):
        x = self.conv(x)
        return F.leaky_relu(x, inplace=True)


class SetBlock(nn.Module):
    def __init__(self, forward_block, pooling=False):
        super(SetBlock, self).__init__()
        self.forward_block = forward_block
        self.pooling = pooling
        if pooling:
            self.pool2d = nn.MaxPool2d(2)
    def forward(self, x):
        n, s, c, h, w = x.size()
        x = self.forward_block(x.view(-1,c,h,w))
        if self.pooling:
            x = self.pool2d(x)
        _, c, h, w = x.size()
        return x.view(n, s, c, h ,w)






class Gateset_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.pcview=PCViews()
        self.hidden_dim=256
        
        _set_in_channels = 1
        _set_channels = [32, 64, 128]
        self.set_layer1 = SetBlock(BasicConv2d(_set_in_channels, _set_channels[0], 5, padding=2))
        self.set_layer2 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[0], 3, padding=1), True)
        self.set_layer3 = SetBlock(BasicConv2d(_set_channels[0], _set_channels[1], 3, padding=1))
        self.set_layer4 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[1], 3, padding=1), True)
        self.set_layer5 = SetBlock(BasicConv2d(_set_channels[1], _set_channels[2], 3, padding=1))
        self.set_layer6 = SetBlock(BasicConv2d(_set_channels[2], _set_channels[2], 3, padding=1))


        _gl_in_channels = 32
        _gl_channels = [64, 128]
        self.gl_layer1 = BasicConv2d(_gl_in_channels, _gl_channels[0], 3, padding=1)
        self.gl_layer2 = BasicConv2d(_gl_channels[0], _gl_channels[0], 3, padding=1)
        self.gl_layer3 = BasicConv2d(_gl_channels[0], _gl_channels[1], 3, padding=1)
        self.gl_layer4 = BasicConv2d(_gl_channels[1], _gl_channels[1], 3, padding=1)
        self.gl_pooling = nn.MaxPool2d(2)

        self.bin_num = [1, 2, 4, 8, 16]
        self.final=nn.Linear(128,self.hidden_dim)


    def frame_max(self, x):
        return torch.max(x, 1)




    def get_img(self,inpt):
        bs=inpt.shape[0]
        imgs=self.pcview.get_img(inpt.permute(0,2,1))
        _,h,w=imgs.shape
        
        imgs=imgs.reshape(bs,6,-1)
        max=torch.max(imgs,-1,keepdim=True)[0]
        min=torch.min(imgs,-1,keepdim=True)[0]
        
        nor_img=(imgs-min)/(max-min)
        nor_img=nor_img.reshape(bs,6,h,w)
        return nor_img



    def forward(self,inpt):
        '''
        norm_img shape is (20,6,128,128)
        20 is the batch_size
        6 is the view number
        128 is the image size
        '''
        norm_img=self.get_img(inpt) # (20,6,128,128)
        norm_img=norm_img.unsqueeze(2)
        
        x=self.set_layer1(norm_img)
        x=self.set_layer2(x)
        gl = self.gl_layer1(self.frame_max(x)[0]) # x's shape is (bs,64,32,32)
        gl = self.gl_layer2(gl) # just normal convolutional network
        gl = self.gl_pooling(gl) # the shape is (40,64,16,16)

        x = self.set_layer3(x) # 40,64,16,16
        x = self.set_layer4(x)
        gl = self.gl_layer3(gl + self.frame_max(x)[0])
        gl = self.gl_layer4(gl)

        x = self.set_layer5(x)
        x = self.set_layer6(x)
        x = self.frame_max(x)[0]
        gl = gl + x

        feature=[]
        n, c, h, w = gl.size()
        for num_bin in self.bin_num:
            z = x.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
            z = gl.view(n, c, num_bin, -1)
            z = z.mean(3) + z.max(3)[0]
            feature.append(z)
        feature = torch.cat(feature, 2).permute(2, 0, 1).contiguous()
        feature=self.final(feature)
        return feature


if __name__=='__main__':
    '''
    5 way
    1 shot
    3 query
    '''

    inpt=torch.randn((20,3,1024))
    network=Gateset_net()
    out=network(inpt)