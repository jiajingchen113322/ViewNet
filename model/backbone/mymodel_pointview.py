import os
import sys
sys.path.append(os.getcwd())

import torch
import torch.nn as nn
from model.backbone.mymodel_moreview import ViewNet
from model.backbone.DGCNN import DGCNN_fs


class pointview(nn.Module):
    def __init__(self):
        super().__init__()
        self.sideview=ViewNet()
        self.pointmodel=DGCNN_fs()
        self.hidden_dim=256
        self.linear=nn.Linear(512,self.hidden_dim)
    
    def forward(self,x):
        view_out=self.sideview(x)
        point_out=self.pointmodel(x)
        
        bin_num=view_out.shape[0]
        point_out_rep=point_out.unsqueeze(0).repeat(bin_num,1,1)
        final_feat=torch.cat((view_out,point_out_rep),-1)
        final_feat=self.linear(final_feat)
        
        return view_out,point_out,final_feat
    
    
if __name__=='__main__':
    inpt=torch.rand((10,3,1024)).cuda()
    
    model=pointview().cuda()
    model(inpt)