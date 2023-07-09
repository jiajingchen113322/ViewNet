# A Novel Projection-Based Backbone with View Pooling for Few-shot Point Cloud Classification (CVPR2023)
This is the pytorch implementation of ViewNet.  
**For a quick learning, you could go to ./model/network.py directly**. I incoporate several backbone and few-shot head into this code, as well as a simple example in it. Feel free to take a look. The ViewNet's Structure is shown below. You could refer to [our paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_ViewNet_A_Novel_Projection-Based_Backbone_With_View_Pooling_for_Few-Shot_CVPR_2023_paper.pdf) for more detail.

![Alt text](https://github.com/jiajingchen113322/ViewNet/blob/master/Img/ViewNet.PNG)


# Arguments
Here are some required arugments you need to give  
```
--exp_name: the experiment name you give  
--dataset: you could fill one of ['ScanObjectNN','ModeNet40','ModeNet40C'].  
--data_path: the path you save the dataset  
--k_way: the number of classes.  
--n_shot: the number of shots.  
--query: the query number  
--fold: The fold number you select to perform testing, for k-fold-cross-validatoin.
```



# ScanObjectNN
## Data Obtaining
We use the point cloud **without background** for the experiment. You could download our preprocessed data directly from [here](https://drive.google.com/file/d/1xzh7a__wHvg6lUAWi-Hbanyt4XHPtw0Y/view?usp=sharing) for using, or you could download the raw data from this [link](https://hkust-vgd.github.io/scanobjectnn/), and put point cloud in different classes in different folder. The dataset folder structure shoule be like shown below:

```
DataSet  
│
└───0
│    |__0.npy
│    |__1.npy
│    |__2.npy
│   
└───1
    │__0.npy
    |__1.npy
    |__2.npy
```

## Run Experiment
Run the following code:
```
python main.py --exp_name exp_scanobjectnn --dataset ScanObjectNN --data_path $path to DataSet$
```
Feel free to change the number of shot,query or fold.  
Your experiment reult will be recorded in ./Exp/ScanObjectNN

# ModelNet40
## Data Obtaining
Similar to ScanObjectNN dataset, you could download our preprocessed data [here](https://drive.google.com/file/d/10faoJ5rRT96Nhdqo9tGD3q7Vg_ZZ2apZ/view?usp=sharing), or you could download the raw data from (here)[https://modelnet.cs.princeton.edu/], and make it a structure like shown above in ScanObjectNN.

## Run Experiment
Run the following code:
```
python main.py --exp_name exp_modelnet40 --dataset ModeNet40 --data_path $path to DataSet$
```
Your experiment reult will be recorded in ./Exp/ModelNet40

# ModelNet40-C
## Data Obatining
You could download our preprocessed data [here](https://drive.google.com/file/d/1EFbGbtmORogjbbQ22giChio3i_G5Oahk/view?usp=sharing), or you could download the raw data [here](https://github.com/jiachens/ModelNet40-C) and make it the stucture shown in ScanobjectNN part. We use the corruption type of "LiDAR", feel free to try different corruption type in raw dataset.
## Run Experiment
```
python main.py --exp_name exp_modelnet40c --dataset ModeNet40C --data_path $path to DataSet$  
```