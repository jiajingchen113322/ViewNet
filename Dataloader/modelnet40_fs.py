import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Sampler
import os
import h5py
# import open3d as o3d
from tqdm import tqdm
from torch.utils.data import DataLoader
np.random.seed(0)


class MiniImageNet(Dataset):
    def __init__(self,root,split='train',num_point=1024,data_aug=True):
        super().__init__()
        self.root=root
        self.split=split
        self.num_point=num_point
        self.data_aug=data_aug

        self.point,self.label=self.get_point()
        
    
    
    def get_point(self):
        point_list=[]
        label_list=[]
        
        for file in os.listdir(self.root):
            file_name,file_format=file.split('.')
            
            if file_format!='h5':
                continue
            
            if file_name.split('_')[0][:-1]==self.split:
                h5_file=h5py.File(os.path.join(self.root,file))
                point=h5_file['data'][:]
                label=h5_file['label'][:]
                
                point_list.append(point)
                label_list.append(label)
                
        point_list=np.concatenate(point_list,0)
        label_list=np.concatenate(label_list,0)
        
        return point_list,label_list.reshape(-1)
    
        
    
    def translate_pointcloud(self,pointcloud):
        xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
        xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
        
        translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
        return translated_pointcloud
    

    
    def __getitem__(self, item):
        pointcloud = self.point[item][:self.num_point]
        label = self.label[item]
        if self.split == 'train' and self.data_aug:
            pointcloud = self.translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        
        pointcloud=torch.FloatTensor(pointcloud)
        label=torch.LongTensor([label])

        pointcloud=pointcloud.permute(1,0)
        return pointcloud, label
    
    
    
    
    def __len__(self):
        return self.point.shape[0]
        






'''
In the WACV paper
- Totoal 80 epochs used for training
- 400 training episodes and 600 validating episodes for each epoch
- For testing, episodes=700
- n_way=5. k_shot=1. query=15 for each classes

'''


class NShotTaskSampler(Sampler):
    def __init__(self,dataset,episode_num,k_way,n_shot,query_num):
        super().__init__(dataset)
        self.dataset=dataset
        self.episode_num=episode_num
        self.k_way=k_way
        self.n_shot=n_shot
        self.query_num=query_num
        self.label_set=self.get_label_set()
        self.data,self.label =self.dataset.point, self.dataset.label
    
    def get_label_set(self):
        point_label_set=np.unique(self.dataset.label)
        return point_label_set
    
    
    
    
    def __iter__(self):
        for _ in range(self.episode_num):
            support_list=[]
            query_list=[]
            picked_cls_set=np.random.choice(self.label_set,self.k_way,replace=False)
            
            for picked_cls in picked_cls_set:
                target_index=np.where(self.label==picked_cls)[0]
                picked_target_index=np.random.choice(target_index,self.n_shot+self.query_num,replace=False)
                
                support_list.append(picked_target_index[:self.n_shot])
                query_list.append(picked_target_index[self.n_shot:])
                
            s=np.concatenate(support_list)
            q=np.concatenate(query_list)
            
            
            '''
            For epi_index
            - it's the index used for each batch
            - the first k_way*n_shot images is the support set
            - the last k_way*query images is for the query set 
            '''    
            epi_index=np.concatenate((s,q))
            yield epi_index
            

    
    
    def __len__(self):
        return self.episode_num
    
        



def get_sets(data_path,k_way=5,n_shot=1,query_num=15,data_aug=False):
    train_dataset=MiniImageNet(root=data_path,split='train',data_aug=data_aug)
    train_sampler=NShotTaskSampler(dataset=train_dataset,episode_num=400,k_way=k_way,n_shot=n_shot,query_num=query_num)
    train_loader=DataLoader(train_dataset,batch_sampler=train_sampler)
    
    val_dataset=MiniImageNet(root=data_path,split='test',data_aug=data_aug)
    val_sampler=NShotTaskSampler(dataset=val_dataset,episode_num=700,k_way=k_way,n_shot=n_shot,query_num=query_num)
    val_loader=DataLoader(val_dataset,batch_sampler=val_sampler)
    
    return train_loader,val_loader






if __name__=='__main__':
    data_path='D:\Computer_vision\Dataset\ModelNet40_fewshot\modelnet40_ply_hdf5_2048_fs'
    train_lod,val_lod=get_sets(data_path=data_path)
    
    dataset=MiniImageNet(root=data_path)
    sampler=NShotTaskSampler(dataset=dataset,episode_num=400,k_way=5,n_shot=1,query_num=5)
    
    
    dataloader = DataLoader(dataset,batch_sampler=sampler,num_workers=0)
    for (x,y) in dataloader:
        '''
        x' shape is (80,3,1024)
        y's shpae is (80,1)
        '''
        pass
    
       
 
    # ========== Data Visulization ===================
    # point,label=dataset[0]
    # point=point.permute(1,0).numpy()
    # print(label)
    
    # pointcloud=o3d.geometry.PointCloud()
    # pointcloud.points=o3d.utility.Vector3dVector(point)
    # o3d.visualization.draw_geometries([pointcloud])
    # =================================================