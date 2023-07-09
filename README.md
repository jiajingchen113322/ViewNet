# A Novel Projection-Based Backbone with View Pooling for Few-shot Point Cloud Classification (CVPR2023)
This is the pytorch implementation of ViewNet.  
**For a quick learning, you could go to ./model/network.py directly**. I incoporate several backbone and few-shot head into this code, as well as a simple example in it. Feel free to take a look. The ViewNet's Structure is shown below. You could refer to the [paper](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_ViewNet_A_Novel_Projection-Based_Backbone_With_View_Pooling_for_Few-Shot_CVPR_2023_paper.pdf) for more detail.

![Alt text](https://github.com/jiajingchen113322/ViewNet/blob/master/Img/ViewNet.PNG)

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

