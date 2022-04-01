import torch
import numpy as np

def cos_sim(a,b,eps=1e-6):
    norm_a,norm_b=torch.norm(a,dim=1),torch.norm(b,dim=1)
    prod_norm=norm_a.unsqueeze(-1)*norm_b.unsqueeze(0)
    prod_norm[prod_norm<eps]=eps

    prod_mat=torch.matmul(a,b.permute(1,0))
    cos_sim=prod_mat/prod_norm

    return cos_sim


if __name__=='__main__':
    a=torch.randn((5,256))
    b=torch.randn((10,256))
    cos_sim(a,b)
