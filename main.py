import torch
from tqdm import tqdm
import argparse
import numpy as np


# from Dataloader.model_net_cross_val import get_sets
# from Dataloader.scanobjectnn_cross_val import get_sets

from util.get_acc import cal_cfm
import torch.nn as nn

# ======== load model =========
from model.network import fs_network


import os
from torch.utils.tensorboard import SummaryWriter
import json
import yaml
import logging

# ============== Get Configuration =================
def get_arg():
    cfg=argparse.ArgumentParser()
    cfg.add_argument('--exp_name',default='try')
    cfg.add_argument('--multigpu',default=False)
    cfg.add_argument('--epochs',default=80,type=int)
    cfg.add_argument('--decay_ep',default=5,type=int)
    cfg.add_argument('--gamma',default=0.7,type=float)
    cfg.add_argument('--lr',default=1e-4,type=float)
    cfg.add_argument('--train',action='store_true',default=True)
    cfg.add_argument('--seed',default=0)
    cfg.add_argument('--device',default='cuda')
    cfg.add_argument('--lr_sch',default=False)
    cfg.add_argument('--data_aug',default=True)
    cfg.add_argument('--dataset',default='ModeNet40C',choices=['ScanObjectNN','ModeNet40','ModeNet40C'])


    # ======== few shot cfg =============#
    cfg.add_argument('--k_way',default=5,type=int)
    cfg.add_argument('--n_shot',default=1,type=int)
    cfg.add_argument('--query',default=10,type=int)
    cfg.add_argument('--backbone',default='ViewNet',choices=['dgcnn','mv','gaitset','ViewNet','Point_Trans'])
    cfg.add_argument('--fs_head',type=str,default='Trip_CIA',choices=['protonet','cia','trip','pv_trip','Trip_CIA','MetaOp','Relation'])
    cfg.add_argument('--fold',default=0)
    # ===================================#


    # ======== path needed ==============#
    cfg.add_argument('--project_path',default=None,help='The path you save this project')
    cfg.add_argument('--data_path',default='D:/Computer_vision/Dataset/ModelNet40_C_fewshot') 
    # ===================================#    
    return cfg.parse_args()


cfg=get_arg()
# ==================================================
if cfg.project_path is None:
    cfg.project_path=os.path.dirname(os.path.abspath(__file__))


if cfg.dataset=='ScanObjectNN':
    cfg.exp_folder_name='ScanObjectNN'
    from Dataloader.scanobjectnn_cross_val import get_sets

elif cfg.dataset=='ModeNet40':
    cfg.exp_folder_name='ModelNet40'
    from Dataloader.model_net_cross_val import get_sets
    
elif cfg.dataset=='ModeNet40C':
    cfg.exp_folder_name='ModelNet40C'
    from Dataloader.model_net_cross_val import get_sets

else:
    raise ValueError('Wrong Dataset Name')

# ============= create logging ==============
def get_logger(file_name='accuracy.log'):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s, %(name)s, %(message)s')

    ########### this is used to set the log file ##########
    exp_file_folder=os.path.join(cfg.project_path,'Exp',cfg.exp_folder_name,cfg.exp_name)
    if not os.path.exists(exp_file_folder):
        os.makedirs(exp_file_folder)
    
    file_handler = logging.FileHandler(os.path.join(exp_file_folder,file_name))
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    #######################################################


    ######### this is used to set the output in the terminal/screen ########
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    #################################################################

    ####### add the log file handler and terminal handerler to the logger #######
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    ##############################################################################

    return logger
# ============================================


def test_model(model,val_loader,cfg):
    global logger
    logger=get_logger(file_name='testing_result.log')

    exp_path=os.path.join(cfg.project_path,cfg.exp_folder_name,cfg.exp_name,'pth_file')
    picked_pth=sorted(os.listdir(exp_path),key=lambda x:int(x.split('_')[-1]))[-1]
    pth_file=torch.load(os.path.join(exp_path,picked_pth))
    model.load_state_dict(pth_file['model_state'])
    

    model=model.cuda()
    bar=tqdm(val_loader,ncols=100,unit='batch',leave=False)
    summary=run_one_epoch(model,bar,'test',loss_func=None)

    acc_list=summary['acc']

    mean_acc=np.mean(acc_list)
    std_acc=np.std(acc_list)

    interval=1.960*(std_acc/np.sqrt(len(acc_list)))
    logger.debug('Mean: {}, Interval: {}'.format(mean_acc*100,interval*100))



def main(cfg):
    global logger
    logger=get_logger()


    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled=False
    
    train_loader,val_loader=get_sets(data_path=cfg.data_path,fold=cfg.fold,k_way=cfg.k_way,n_shot=cfg.n_shot,query_num=cfg.query,data_aug=cfg.data_aug)
    model=fs_network(k_way=cfg.k_way,n_shot=cfg.n_shot,query=cfg.query,backbone=cfg.backbone,fs=cfg.fs_head)
    if cfg.multigpu:
        model=nn.DataParallel(model)
    
    if cfg.train:
        train_model(model,train_loader,val_loader,cfg)
    
    else:
        test_model(model,val_loader,cfg)
    


def train_model(model,train_loader,val_loader,cfg):
    device=torch.device(cfg.device)
    model=model.to(device)
    
    #====== loss and optimizer =======
    loss_func=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.parameters(),lr=cfg.lr)
    if cfg.lr_sch:
        lr_schedule=torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=np.arange(10,cfg.epochs,cfg.decay_ep),gamma=cfg.gamma)
    
    
    def train_one_epoch():
        bar=tqdm(train_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model,bar,'train',loss_func=loss_func,optimizer=optimizer)
        summary={"loss/train":np.mean(epsum['loss'])}
        return summary
        
        
    def eval_one_epoch():
        bar=tqdm(val_loader,ncols=100,unit='batch',leave=False)
        epsum=run_one_epoch(model,bar,"valid",loss_func=loss_func)
        mean_acc=np.mean(epsum['acc'])
        summary={'meac':mean_acc}
        summary["loss/valid"]=np.mean(epsum['loss'])
        return summary,epsum['cfm'],epsum['acc']
    
    
    # ======== define exp path ===========
    exp_path=os.path.join(cfg.project_path,'Exp',cfg.exp_folder_name,cfg.exp_name)
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)


    # save config into json #
    cfg_dict=vars(cfg)
    yaml_file=os.path.join(exp_path,'config.yaml')
    with open(yaml_file,'w') as outfile:
        yaml.dump(cfg_dict, outfile, default_flow_style=False)
    # f = open(json_file, "w")
    # json.dump(cfg_dict, f)
    # f.close()
    #########################
    
    tensorboard=SummaryWriter(log_dir=os.path.join(exp_path,'TB'),purge_step=cfg.epochs)
    pth_path=os.path.join(exp_path,'pth_file')
    if not os.path.exists(pth_path):
        os.mkdir(pth_path)
    # =====================================
    
    # ========= train start ===============
    acc_list=[]
    interval_list=[]

    tqdm_epochs=tqdm(range(cfg.epochs),unit='epoch',ncols=100)
    for e in tqdm_epochs:
        train_summary=train_one_epoch()
        val_summary,conf_mat,batch_acc_list=eval_one_epoch()
        summary={**train_summary,**val_summary}
        
        if cfg.lr_sch:
            lr_schedule.step()
        
        accuracy=val_summary['meac']
        acc_list.append(val_summary['meac'])

        # === get 95% interval =====
        std_acc=np.std(batch_acc_list)
        interval=1.960*(std_acc/np.sqrt(len(batch_acc_list)))
        interval_list.append(interval)

        max_acc_index=np.argmax(acc_list)
        max_ac=acc_list[max_acc_index]
        max_interval=interval_list[max_acc_index]
        # ===========================

        logger.debug('epoch {}: {}. Highest: {}. Interval: {}'.format(e,accuracy,max_ac,max_interval))
        # print('epoch {}: {}. Highese: {}'.format(e,accuracy,np.max(acc_list)))
        
        if np.max(acc_list)==acc_list[-1]:
            summary_saved={**summary,
                            'model_state':model.state_dict(),
                            'optimizer_state':optimizer.state_dict(),
                            'cfm':conf_mat}
            torch.save(summary_saved,os.path.join(pth_path,'epoch_{}'.format(e)))
        
        for name,val in summary.items():
            tensorboard.add_scalar(name,val,e)
    
    summary_saved={**summary,
                'model_state':model.module.state_dict(),
                'optimizer_state':optimizer.state_dict(),
                'cfm':conf_mat,
                'acc_list':acc_list}
    torch.save(summary_saved,os.path.join(pth_path,'epoch_final'))



    # =======================================    
    
    



def run_one_epoch(model,bar,mode,loss_func,optimizer=None,show_interval=10):
    confusion_mat=np.zeros((cfg.k_way,cfg.k_way))
    summary={"acc":[],"loss":[]}
    device=next(model.parameters()).device
    
    if mode=='train':
        model.train()
    else:
        model.eval()
    
    for i, (x_cpu,y_cpu) in enumerate(bar):
        x,y=x_cpu.to(device),y_cpu.to(device)
        
        if mode=='train':
            optimizer.zero_grad()
            pred,loss=model(x)
            
            #==take one step==#
            loss.backward()
            optimizer.step()
            #=================#
        else:
            with torch.no_grad():
                pred,loss=model(x)
        
        
        summary['loss']+=[loss.item()]
        
        if mode=='train':
            if i%show_interval==0:
                bar.set_description("Loss: %.3f"%(np.mean(summary['loss'])))
        else:
            batch_cfm=cal_cfm(pred,model.q_label,ncls=cfg.k_way)
            batch_acc=np.trace(batch_cfm)/np.sum(batch_cfm)
            summary['acc'].append(batch_acc)
            if i%show_interval==0:
                bar.set_description("mea_ac: %.3f"%(np.mean(summary['acc'])))
            
            confusion_mat+=batch_cfm
    
    if mode!='train':
        summary['cfm']=confusion_mat
    
    return summary
            



if __name__=='__main__':
    main(cfg)
    