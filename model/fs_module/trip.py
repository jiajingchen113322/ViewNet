import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class trip(nn.Module):
    def __init__(self,k_way,n_shot,query):
        super().__init__()
        self.k=k_way
        self.n=n_shot
        self.q=query
        self.margin=0.2


    def batch_dist(self,feat):
        return torch.cdist(feat,feat)



    def trip_loss(self,feature,label):
        # ==== get label ====
        bin_num,sample_num,fd=feature.shape
        label=label.unsqueeze(0).repeat(bin_num,1)
        label=label.to(feature.device)
        # ===================

        # ==== get mask and dist ====
        hp_mask = (label.unsqueeze(1) == label.unsqueeze(2)).bool().view(-1)
        hn_mask = (label.unsqueeze(1) != label.unsqueeze(2)).bool().view(-1)
        dist=self.batch_dist(feature)
        dist=dist.view(-1)
        # ===========================

        full_hp_dist=torch.masked_select(dist,hp_mask).reshape(bin_num,sample_num,-1,1)
        full_hn_dist=torch.masked_select(dist,hn_mask).reshape(bin_num,sample_num,1,-1)
        full_loss_metric=F.relu(self.margin+full_hp_dist-full_hn_dist).view(bin_num,-1)
        
        full_loss_metric_sum=torch.sum(full_loss_metric,1)
        full_loss_num=(full_loss_metric!=0).sum(1).float()
        full_loss_mean=full_loss_metric_sum/full_loss_num
        full_loss_mean[full_loss_num == 0] = 0
        
        return full_loss_mean.mean()



    def getout(self,support,queries):
        dist=torch.cdist(queries,support)
        dist=torch.mean(dist,0)
        y_pred=(-dist).softmax(1)
        return y_pred




    def forward(self,inpt,label):
        label=torch.cat(label)

        # support (62,5,256)
        # query (62,15,256)
        bin_num,_,fd=inpt.shape
        support=inpt[:,:self.k*self.n,:].reshape(bin_num,self.k,self.n,fd)
        support=torch.mean(support,2)
        queries=inpt[:,self.k*self.n:,:]

        loss=self.trip_loss(torch.cat((support,queries),1),label)
        y_pred=self.getout(support,queries)
        return y_pred,loss






if __name__=='__main__':
    '''
    If backbone is the gait related network
    the embeding shape is (bin,sample_num,feat_dim), like (62,20,256)
    '''
    k=5
    q=3
    n=1

    inpt=torch.randn((62,20,256))
    query_label=torch.arange(k).repeat_interleave(3)
    sup_label=torch.arange(k)
    fs=trip(k_way=k,n_shot=n,query=q)
    fs(inpt,[sup_label,query_label])
   