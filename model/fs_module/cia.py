import os
import sys
sys.path.append(os.getcwd())

import torch
import numpy as np
import torch.nn as nn
from util.dist import cos_sim


class protonet(nn.Module):
    def __init__(self,k_way,n_shot,query):
        super().__init__()
        self.loss_fn=torch.nn.NLLLoss()
        
        self.k=k_way
        self.n=n_shot
        self.query=query
       

    
    def get_dist(self,feat):
        support=feat[:self.n*self.k]
        queries=feat[self.n*self.k:]
        
        prototype=support.reshape(self.k,self.n,-1).mean(1)
        distance=torch.cdist(queries.unsqueeze(0),prototype.unsqueeze(0)).squeeze(0)
        return distance
    
    def forward(self,feat,label):
        dist=self.get_dist(feat)
        y_pred=(-dist).softmax(1)
        
        log_p_y = (-dist).log_softmax(dim=1)
        loss = self.loss_fn(log_p_y, label.to(feat.device))
        
        return y_pred,loss
    

        



class CIA(nn.Module):
    def __init__(self,k_way,n_shot,query,feat_dim=256):
        super().__init__()
        self.loss_fn=torch.nn.NLLLoss()
        
        self.k=k_way
        self.n=n_shot
        self.query=query
        self.to_qk=nn.Linear(feat_dim,2*feat_dim)
        self.feat_dim=feat_dim
        self.K1=13
        self.K2=2

        self.support_weight=nn.Sequential(nn.Linear(self.K1+1,16),
                                          nn.ReLU(),
                                          nn.Linear(16,self.K1+1),
                                          nn.Softmax(-1))
        
        self.query_weight=nn.Sequential(nn.Linear(self.K2+1,16),
                                        nn.ReLU(),
                                        nn.Linear(16,self.K2+1),
                                        nn.Softmax(-1))
        


    def self_interaction(self,v):
        feat=self.to_qk(v)
        q,k=torch.split(feat,self.feat_dim,-1)

        # === get R matrix ===
        R_mat=torch.bmm(q.unsqueeze(-1),k.unsqueeze(1))
        R_mat=nn.functional.softmax(R_mat,1)
        # ====================

        final_feat=torch.bmm(v.unsqueeze(1),R_mat).squeeze(1)+v
        return final_feat


    def cross_fusion(self,feat):
        support=feat[:self.n*self.k]
        queries=feat[self.n*self.k:]

        # dist=torch.cdist(support,queries)
        sim=cos_sim(support,queries)
        dist=-sim

        # == obtain fused support ==
        index=torch.argsort(dist,-1)[:,:self.K1]
        support_num,_=index.shape
        cat_support=queries[index.reshape(-1),:].reshape(support_num,self.K1,-1)
        cat_support=torch.cat([support.unsqueeze(1),cat_support],1).permute(0,2,1)
        support_weight=self.support_weight(cat_support)
        support_feat=torch.sum(cat_support*support_weight,-1)
        # ===========================

        # == obtain fused query ==
        dist=dist.permute(1,0)
        index=torch.argsort(dist,-1)[:,:self.K2]
        query_num,_=index.shape
        cat_query=support[index.reshape(-1),:].reshape(query_num,self.K2,-1)
        cat_query=torch.cat([queries.unsqueeze(1),cat_query],1).permute(0,2,1)
        query_weight=self.query_weight(cat_query)
        query_feat=torch.sum(query_weight*cat_query,-1)
        # =========================

        return support_feat,query_feat



    def get_dist(self,support,queries):
        # support=feat[:self.n*self.k]
        # queries=feat[self.n*self.k:]
        
        prototype=support.reshape(self.k,self.n,-1).mean(1)
        distance=torch.cdist(queries.unsqueeze(0),prototype.unsqueeze(0)).squeeze(0)
        return distance
        

    def forward(self,sample_inpt,label):
        feat=self.self_interaction(sample_inpt)
        support_feat,query_feat =self.cross_fusion(feat)

        dist=self.get_dist(support_feat,query_feat)
        y_pred=(-dist).softmax(1)
        log_p_y = (-dist).log_softmax(dim=1)
        loss = self.loss_fn(log_p_y, label[1].to(feat.device))
        
        return y_pred,loss
    


    



if __name__=='__main__':
    k_way=5
    query=3
    shot=1

    sample_inpt=torch.randn((30,1024))
    label=torch.arange(k_way).repeat_interleave(query)
    # net=protonet(k_way=k_way,n_shot=shot,query=query)
    # pred,loss= net(sample_inpt,label)
    net=CIA(k_way=k_way,n_shot=shot,query=query)
    net(sample_inpt,label)

    a=1
    
    