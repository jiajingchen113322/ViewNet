import numpy as np
import open3d as o3d
import os
import sys
sys.path.append(os.getcwd())

from tqdm import tqdm
import pywavefront
from util.obj_io import parse_obj_file
# trash=['03001627/42db4f765f4e6be414038d588fd1342f','03001627/bee929929700e99fad8a0ee9b106700e']
abnormal_list=[]

def build_point(path,target_path):
    data_list=os.listdir(path)
    data_list=[i for i in data_list if i[-3:]!='son']
    
    for cls in tqdm(data_list):
        cls_path=os.path.join(path,cls)
        
        # === make the path to which cls is saved to ===
        save_cls_path=os.path.join(target_path,cls)
        if not os.path.exists(save_cls_path):
            os.mkdir(save_cls_path)
        # =============================================

        sample_list=os.listdir(cls_path)
        for s in tqdm(sample_list,leave=False):
            sample_path=os.path.join(cls_path,s,'models/model_normalized.obj')
            
            try:
            # ===== create open3d triangle mesh =====
                obj=pywavefront.Wavefront(sample_path,collect_faces=True)
                face=np.array(obj.mesh_list[0].faces)
                vertices=np.array(obj.vertices)
                mesh=o3d.geometry.TriangleMesh()
                mesh.vertices=o3d.utility.Vector3dVector(vertices)
                mesh.triangles=o3d.utility.Vector3iVector(face)
                pcd=mesh.sample_points_uniformly(number_of_points=1024)
               
            except:
                print(sample_path)
                abnormal_list.append(sample_path)
                # mesh = o3d.io.read_triangle_mesh(sample_path)
                # pcd=mesh.sample_points_uniformly(number_of_points=1024)
                continue

            points=np.array(pcd.points)
            sample_save_path=os.path.join(save_cls_path,s)
            np.save(sample_save_path,points)
        
    np.save('abormal_path_list',abnormal_list)




def recycle_file(target_path):
    abnormal_list = '/data1/jiajing/worksapce/My_Research/few_shot_point/Few_Shot_Point_Cloud/util/abormal_path_list.npy'
    abnormal_list=np.load(abnormal_list)
    invalid_list=[]

    for i in tqdm(list(abnormal_list)):
        # ==== get saving path ====
        abn=str(i)
        class_id,sample_id=abn.split('/')[-4],abn.split('/')[-3]
        save_path=os.path.join(target_path,class_id,sample_id)
        if os.path.exists(save_path+'.npy'):
            continue
        # =========================
        try:
        # ==== get point =====
            obj_file=parse_obj_file(open(abn))
            vertices,face=obj_file[:2]
            mesh=o3d.geometry.TriangleMesh()
            mesh.vertices=o3d.utility.Vector3dVector(vertices)
            mesh.triangles=o3d.utility.Vector3iVector(face)
            pcd=mesh.sample_points_uniformly(number_of_points=1024)
            points=np.array(pcd.points)
            np.save(save_path,points)
        except:
            invalid_list.append(abn)
    
    print("invalid number is ",len(invalid_list))
    np.save('invalid_list',invalid_list)


if __name__=='__main__':
    data_path='/data1/minmin/shapenet/ShapeNetCore.v2'
    target_path='/data1/jiajing/dataset/ShapeNet'
    
    
    # s='/data1/minmin/shapenet/ShapeNetCore.v2/03001627/1842797b090ce2ecebc1a7ae7c4c250d/models/model_normalized.obj'

    # ==== build npy point by open3d or pywavefront ====
    # build_point(data_path,target_path)
    # ==================================================


    # ==== recyle the file unable to be loaded by open3d or pywavefront ===
    recycle_file(target_path)

    # =====================================================================

