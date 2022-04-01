import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class pointview_trip(nn.Module):
    def __init__(self,k_way,n_shot,query):
        super().__init__()
        self.k=k_way
        self.n=n_shot
        self.q=query
        self.margin=0.2
        
    def trip_loss(self,feature,label):
        # ==== get feat ===
        bin_num,_,fd=feature.shape 
        support=feature[:,:self.k*self.n,:].reshape(bin_num,self.k,self.n,fd)
        support=torch.mean(support,2)
        queries=feature[:,self.k*self.n:,:]
        feature=torch.cat((support,queries),1)
        # =================
        
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

    def batch_dist(self,feat):
        return torch.cdist(feat,feat)
    
    
    def getout(self,feature):
        bin_num,_,fd=feature.shape 
        support=feature[:,:self.k*self.n,:].reshape(bin_num,self.k,self.n,fd)
        support=torch.mean(support,2)
        queries=feature[:,self.k*self.n:,:]
        
        dist=torch.cdist(queries,support)
        dist=torch.mean(dist,0)
        y_pred=(-dist).softmax(1)
        return y_pred
    
    
    
    
    def forward(self,x,label):
        view_out,point_out,final_out=x
        bin_num=view_out.shape[0]
        point_out=point_out.unsqueeze(0).repeat(bin_num,1,1) # repeated point out
        label=torch.cat(label)

        view_loss=self.trip_loss(view_out,label)
        point_loss=self.trip_loss(point_out,label)
        feat_loss=self.trip_loss(final_out,label)
        
        loss=view_loss+point_loss+feat_loss
        y_pred=self.getout(final_out)
        return y_pred,loss
        
        
        
        
        
        
    
    
    
if __name__=='__main__':
    
    # === config ====
    k=5
    q=3
    n=1
    # ===============
    
    
    view_out=torch.randn((62,20,256))
    point_out=torch.randn((20,256))
    final_out=torch.randn((62,20,256))
    inpt=[view_out,point_out,final_out]
    
    sup_label=torch.arange(k)
    query_label=torch.arange(k).repeat_interleave(q)
    
    network=pointview_trip(k,n,q)
    out=network(inpt,[sup_label,query_label])