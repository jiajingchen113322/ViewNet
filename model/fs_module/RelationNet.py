import enum
import torch
from torch import nn
from torch import optim
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np


# class RelationNet(nn.Module):
#     """
#         concat -> layer 4 layer 5 avg pooling -> fc -> sigmoid. 
#     """
#     def __init__(self, k_way, n_shot, query) -> None:
#         super(RelationNet, self).__init__()
#         self.k_way = k_way
#         self.n_shot = n_shot
#         self.query = query


#         self.g = nn.Sequential(
#             nn.Linear(512, 256, bias=False),
#             nn.BatchNorm1d(256),
#             nn.ReLU()
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(256, 64),
#             nn.BatchNorm1d(64),
#             nn.ReLU(),
#             nn.Linear(64, 1),
#             nn.Sigmoid()
#         )




#     def forward(self,feat, label):
#         """
#             x is the feat from backbone.  [setsz+querysz, 256]
#             setsz = self.k_way*self.n_shot
#             querysz = self.k_way*self.query
#             batchsz = 1, omit
#         """
#         support_xf=feat[:self.n_shot*self.k_way] # [setsz, 256]
#         query_xf=feat[self.n_shot*self.k_way:]  # [querysz, 256]
#         support_y = label[0]
#         query_y = label[1]

#         setsz = support_xf.size(0)
#         querysz  = query_xf.size(0)

#         # concat each query x with all sets along channel. 
#         support_xf = support_xf.unsqueeze(0).expand(querysz, -1, -1) # [querysz, setsz, c]
#         query_xf = query_xf.unsqueeze(1).expand(-1, setsz, -1) #[querysz, setsz, c]
#         comb = torch.cat([support_xf, query_xf], dim=2) # [querysz, setsz, 2c]

#         # G_phi network
#         comb = self.g(comb.view(querysz*setsz,-1))
#         score = self.fc(comb).squeeze(-1).view(querysz, setsz) # [querysz, setsz]

#         # build its label
#         support_yf = support_y.unsqueeze(0).expand(querysz, setsz)
#         query_yf = query_y.unsqueeze(1).expand(querysz, setsz)
#         label = torch.eq(support_yf, query_yf).float().cuda()

#         # score: [1, querysz, setsz]
#         # label: [1, querysz, setsz]

#         loss = torch.pow(label-score, 2).sum()

#         if self.training:
#             return None, loss

#         # else:
#         rn_score_np = score.cpu().data.numpy()
#         pred = []
#         support_y_np = support_y.cpu().data.numpy()
#         for query in rn_score_np:
#             # query [setsz]
#             sim = []
#             for way in range(self.k_way):
#                 sim.append(np.sum(query[way*self.n_shot : (way+1)*self.n_shot]))
#             idx = np.array(sim).argmax()
#             pred.append(support_y_np[idx*self.n_shot])

#         # pred: [querysz]
#         pred = torch.from_numpy(np.array(pred))
#         correct = torch.eq(pred, query_y).sum().item()
#         total_num = query_y.size(0)
#         return (correct, total_num), loss

class RelationNet(nn.Module):
    """
        refer from : https://github.com/floodsung/LearningToCompare_FSL    
    """
    feature_dim = 256
    def __init__(self, k_way, n_shot, query) -> None:
            super(RelationNet, self).__init__()
            self.class_num = k_way
            self.sample_num_per_class = n_shot
            self.batch_num_per_class = query

            self.fc1 = nn.Sequential(
                nn.Linear(self.feature_dim*2, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
            )
            self.fc2 = nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64,8),
                nn.ReLU(),
                nn.Linear(8,1),
                nn.Sigmoid()
            )

            self.mse = nn.MSELoss()
           
    def forward(self,feat, label): # [B, -1]
        """
    #             x is the feat from backbone.  [setsz+querysz, 256]
    #             setsz = self.k_way*self.n_shot
    #             querysz = self.k_way*self.query
    #             batchsz = 1, omit
    #       """
        sample_features=feat[:self.class_num*self.sample_num_per_class] # [setsz, 256]  -> 25*256
        sample_features = sample_features.view(self.class_num, self.sample_num_per_class, self.feature_dim)
        sample_features = torch.sum(sample_features, 1).squeeze(1) # [class_num, self.feature_dim] -> 5*256
        batch_features=feat[self.class_num*self.sample_num_per_class:]  # [querysz, 256] -> [15*256]

        sample_labels = label[0]
        batch_labels = label[1]

        sample_features_ext = sample_features.unsqueeze(0).repeat(self.batch_num_per_class*self.class_num,1,1)  # (15, 5, 256)
        batch_features_ext = batch_features.unsqueeze(0).repeat(self.class_num, 1, 1)                           # (5, 15, 256)
        batch_features_ext = torch.transpose(batch_features_ext, 0, 1)                                          # (15, 5, 256)
        relation_pairs = torch.cat((sample_features_ext, batch_features_ext), 2).view(-1, self.feature_dim*2)  # [k_way*n_shot*query, feat*2]

        relations = self.fc1(relation_pairs)
        relations = self.fc2(relations)
        relations = relations.view(-1, self.class_num)


        one_hot_labels = torch.zeros(self.batch_num_per_class*self.class_num, self.class_num).scatter_(1, batch_labels.view(-1, 1), 1).to(feat.device)
        loss = self.mse(relations, one_hot_labels)

        # _, pred = torch.max(relations.data, 1)
        return relations, loss




if __name__=='__main__':
    sample_inpt=torch.randn((40,256))
    net=RelationNet(k_way=5,n_shot=5,query=3)
    s_label=torch.arange(5)
    q_label=torch.arange(5).repeat_interleave(3)
    pred,loss= net(sample_inpt,[s_label,q_label])