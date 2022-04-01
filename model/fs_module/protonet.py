import torch
import numpy as np
import torch.nn as nn



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
        loss = self.loss_fn(log_p_y, label[1].to(feat.device))
        
        return y_pred,loss
    
    # def get_loss(self,feat):
    #     dist=self.get_dist(feat)
    #     log_p_y = (-dist).log_softmax(dim=1)
    #     loss = self.loss_fn(log_p_y, self.label.to(feat.device))
    #     return loss
        
        
    



if __name__=='__main__':
    sample_inpt=torch.randn((20,512))
    net=protonet(k_way=5,n_shot=1,query=3)
    label=torch.arange(5).repeat_interleave(3)
    pred,loss= net(sample_inpt,label)
    
    