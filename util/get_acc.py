import torch
import numpy as np
from sklearn.metrics import confusion_matrix

def cal_cfm(pred,label,ncls):
    pred=pred.cpu().detach().numpy()
    label=label.cpu().detach().numpy()
    
    pred=np.argmax(pred,1)
    cfm=confusion_matrix(label,pred,labels=np.arange(ncls))
    return cfm