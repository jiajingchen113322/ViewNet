import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
import torch.nn as nn
from util.dist import cos_sim
import torch.nn.functional as F





class contrastive_bin(nn.Module):
    def __init__(self,k_way,n_shot,query,feat_dim=256):
        super().__init__()
        self.k=k_way
        self.n=n_shot
        self.query=query
        self.loss_fn=torch.nn.NLLLoss()
    
    def forward(self,sample_inpt,label):
        bin_num,_,fd=sample_inpt.shape
        support=sample_inpt[:,:self.k*self.n,:].reshape(bin_num,self.k,self.n,fd)
        support=torch.mean(support,2)
        queries=sample_inpt[:,self.k*self.n:,:]
        
        dist=torch.cdist(queries,support)
        dist=torch.mean(dist,0)
        y_pred=(-dist).softmax(1)

        log_p_y = (-dist).log_softmax(dim=1)
        loss = self.loss_fn(log_p_y, label[1].to(sample_inpt.device))
        return y_pred,loss


class Trip_CIA(nn.Module):
    def __init__(self,k_way,n_shot,query,feat_dim=256):
        super().__init__()
        self.k=k_way
        self.n=n_shot
        self.query=query
        self.feat_dim=feat_dim
        self.k1=13
        self.k2=2
        self.margin=0.2
        # ==== network ====
        self.to_qk=nn.Linear(feat_dim,2*feat_dim)
        self.support_weight=nn.Sequential(nn.Linear(self.k1+1,16),
                                          nn.ReLU(),
                                          nn.Linear(16,self.k1+1),
                                          nn.Softmax(-1))
    
        self.query_weight=nn.Sequential(nn.Linear(self.k2+1,16),
                                        nn.ReLU(),
                                        nn.Linear(16,self.k2+1),
                                        nn.Softmax(-1))




    def self_interaction(self,v):
        feat=self.to_qk(v)
        q,k=torch.split(feat,self.feat_dim,-1)

        # === get R matrix ====
        R_mat=torch.einsum('bijk,bikx->bijx',q.unsqueeze(-1),k.unsqueeze(-2))
        R_mat=nn.functional.softmax(R_mat,1)
        # =====================

        final_feat=torch.einsum('bijk,bikx->bijx',v.unsqueeze(2),R_mat)
        final_feat=final_feat.squeeze(2)+v
        return final_feat


    def get_bin_sim(self,a,b,eps=1e-6):
        norm_a,norm_b=torch.norm(a,dim=-1),torch.norm(b,dim=-1)
        prod_norm=norm_a.unsqueeze(-1)*norm_b.unsqueeze(1)
        prod_norm[prod_norm<eps]=eps

        prod_mat=torch.bmm(a,b.permute(0,2,1))
        cos_sim=prod_mat/prod_norm
        return cos_sim

    def cross_fusion(self,feat):
        bin_num=feat.shape[0]
        support=feat[:,:self.n*self.k,:]
        quries=feat[:,self.n*self.k:,:]
        
        # === get distance ===
        bin_sim=self.get_bin_sim(support,quries)
        dist=-bin_sim
        # ====================

        # === obtain fused support ====
        index=torch.argsort(dist,-1)[:,:,:self.k1]
        support_num=index.shape[1]
        neigbor_index=index.reshape(bin_num,-1).unsqueeze(-1).repeat(1,1,self.feat_dim)
        neighbor_feat=quries.gather(1,neigbor_index)
        neighbor_feat=neighbor_feat.reshape(bin_num,support_num,-1,self.feat_dim)
        cat_support=torch.cat((support.unsqueeze(2),neighbor_feat),2).permute(0,1,3,2)
        support_weight=self.support_weight(cat_support)
        support_feat=torch.sum(cat_support*support_weight,-1)
        # =============================

        # ==== obtain fused query ======
        dist=dist.permute(0,2,1)
        index=torch.argsort(dist,-1)[:,:,:self.k2]
        query_num=index.shape[1]
        neigbor_index=index.reshape(bin_num,-1).unsqueeze(-1).repeat(1,1,self.feat_dim)
        neighbor_feat=support.gather(1,neigbor_index)
        neighbor_feat=neighbor_feat.reshape(bin_num,query_num,-1,self.feat_dim)
        cat_query=torch.cat((quries.unsqueeze(2),neighbor_feat),2).permute(0,1,3,2)
        query_weight=self.query_weight(cat_query)
        query_feat=torch.sum(cat_query*query_weight,-1)
        # ===============================

        return support_feat,query_feat

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


        feat=self.self_interaction(inpt)
        support_feat,query_feat=self.cross_fusion(feat)
        
        # ==== get prototype ====
        bin_num,_,fd=support_feat.shape
        support=support_feat[:,:self.k*self.n,:].reshape(bin_num,self.k,self.n,fd)
        support=torch.mean(support,2)
        # =======================

        loss=self.trip_loss(torch.cat((support,query_feat),1),label)
        y_pred=self.getout(support,query_feat)
        return y_pred,loss




if __name__=='__main__':
    k_way=5
    query=3
    shot=1

    sample_inpt=torch.randn((62,20,256))
    label=[torch.arange(k_way),torch.arange(k_way).repeat_interleave(query)]
    # net=protonet(k_way=k_way,n_shot=shot,query=query)
    # pred,loss= net(sample_inpt,label)
    net=Trip_CIA(k_way=k_way,n_shot=shot,query=query)
    net(sample_inpt,label)

    a=1