import torch
from torch.autograd import Variable
import torch.nn as nn
from qpth.qp import QPFunction
import torch.nn.functional as F

def computeGramMatrix(A, B):
    """
    Constructs a linear kernel matrix between A and B.
    We assume that each row in A and B represents a d-dimensional feature vector.
    
    Parameters:
      A:  a (n_batch, n, d) Tensor.
      B:  a (n_batch, m, d) Tensor.
    Returns: a (n_batch, n, m) Tensor.
    """
    
    assert(A.dim() == 3)
    assert(B.dim() == 3)
    assert(A.size(0) == B.size(0) and A.size(2) == B.size(2))

    return torch.bmm(A, B.transpose(1,2))

def batched_kronecker(matrix1, matrix2):
    matrix1_flatten = matrix1.reshape(matrix1.size()[0], -1)
    matrix2_flatten = matrix2.reshape(matrix2.size()[0], -1)
    return torch.bmm(matrix1_flatten.unsqueeze(2), matrix2_flatten.unsqueeze(1)).reshape([matrix1.size()[0]] + list(matrix1.size()[1:]) + list(matrix2.size()[1:])).permute([0, 1, 3, 2, 4]).reshape(matrix1.size(0), matrix1.size(1) * matrix2.size(1), matrix1.size(2) * matrix2.size(2))


def one_hot(indices, depth):
    """
    Returns a one-hot tensor.
    This is a PyTorch equivalent of Tensorflow's tf.one_hot.
        
    Parameters:
      indices:  a (n_batch, m) Tensor or (m) Tensor.
      depth: a scalar. Represents the depth of the one hot dimension.
    Returns: a (n_batch, m, depth) Tensor or (m, depth) Tensor.
    """

    encoded_indicies = torch.zeros(indices.size() + torch.Size([depth])).cuda()
    index = indices.view(indices.size()+torch.Size([1])).cuda()
    encoded_indicies = encoded_indicies.scatter_(1,index,1)
    
    return encoded_indicies


def MetaOptNetHead_SVM_CS(query, support, support_labels, n_way, n_shot, C_reg=0.1, double_precision=False, maxIter=15):
    """
    Fits the support set with multi-class SVM and 
    returns the classification score on the query set.
    
    This is the multi-class SVM presented in:
    On the Algorithmic Implementation of Multiclass Kernel-based Vector Machines
    (Crammer and Singer, Journal of Machine Learning Research 2001).
    This model is the classification head that we use for the final version.
    Parameters:
      query:  a (tasks_per_batch, n_query, d) Tensor.
      support:  a (tasks_per_batch, n_support, d) Tensor.
      support_labels: a (tasks_per_batch, n_support) Tensor.
      n_way: a scalar. Represents the number of classes in a few-shot classification task.
      n_shot: a scalar. Represents the number of support examples given per class.
      C_reg: a scalar. Represents the cost parameter C in SVM.
    Returns: a (tasks_per_batch, n_query, n_way) Tensor.
    """
    
    tasks_per_batch = query.size(0)
    n_support = support.size(1)
    n_query = query.size(1)

    assert(query.dim() == 3)
    assert(support.dim() == 3)
    assert(query.size(0) == support.size(0) and query.size(2) == support.size(2))
    assert(n_support == n_way * n_shot)      # n_support must equal to n_way * n_shot

    #Here we solve the dual problem:
    #Note that the classes are indexed by m & samples are indexed by i.
    #min_{\alpha}  0.5 \sum_m ||w_m(\alpha)||^2 + \sum_i \sum_m e^m_i alpha^m_i
    #s.t.  \alpha^m_i <= C^m_i \forall m,i , \sum_m \alpha^m_i=0 \forall i

    #where w_m(\alpha) = \sum_i \alpha^m_i x_i,
    #and C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    #This borrows the notation of liblinear.
    
    #\alpha is an (n_support, n_way) matrix
    kernel_matrix = computeGramMatrix(support, support)

    id_matrix_0 = torch.eye(n_way).expand(tasks_per_batch, n_way, n_way).cuda()
    block_kernel_matrix = batched_kronecker(kernel_matrix, id_matrix_0)
    #This seems to help avoid PSD error from the QP solver.
    block_kernel_matrix += 10.0 * torch.eye(n_way*n_support).expand(tasks_per_batch, n_way*n_support, n_way*n_support).cuda()
    
    support_labels_one_hot = one_hot(support_labels.view(tasks_per_batch * n_support), n_way) # (tasks_per_batch * n_support, n_support)
    support_labels_one_hot = support_labels_one_hot.view(tasks_per_batch, n_support, n_way)
    support_labels_one_hot = support_labels_one_hot.reshape(tasks_per_batch, n_support * n_way)
    
    G = block_kernel_matrix
    e = -1.0 * support_labels_one_hot
    #print (G.size())
    #This part is for the inequality constraints:
    #\alpha^m_i <= C^m_i \forall m,i
    #where C^m_i = C if m  = y_i,
    #C^m_i = 0 if m != y_i.
    id_matrix_1 = torch.eye(n_way * n_support).expand(tasks_per_batch, n_way * n_support, n_way * n_support)
    C = Variable(id_matrix_1)
    h = Variable(C_reg * support_labels_one_hot)
    #print (C.size(), h.size())
    #This part is for the equality constraints:
    #\sum_m \alpha^m_i=0 \forall i
    id_matrix_2 = torch.eye(n_support).expand(tasks_per_batch, n_support, n_support).cuda()

    A = Variable(batched_kronecker(id_matrix_2, torch.ones(tasks_per_batch, 1, n_way).cuda()))
    b = Variable(torch.zeros(tasks_per_batch, n_support))
    #print (A.size(), b.size())
    if double_precision:
        G, e, C, h, A, b = [x.double().cuda() for x in [G, e, C, h, A, b]]
    else:
        G, e, C, h, A, b = [x.float().cuda() for x in [G, e, C, h, A, b]]

    # Solve the following QP to fit SVM:
    #        \hat z =   argmin_z 1/2 z^T G z + e^T z
    #                 subject to Cz <= h
    # We use detach() to prevent backpropagation to fixed variables.
    qp_sol = QPFunction(verbose=False, maxIter=maxIter)(G, e.detach(), C.detach(), h.detach(), A.detach(), b.detach())

    # Compute the classification score.
    compatibility = computeGramMatrix(support, query)
    compatibility = compatibility.float()
    compatibility = compatibility.unsqueeze(3).expand(tasks_per_batch, n_support, n_query, n_way)
    qp_sol = qp_sol.reshape(tasks_per_batch, n_support, n_way)
    logits = qp_sol.float().unsqueeze(2).expand(tasks_per_batch, n_support, n_query, n_way)
    logits = logits * compatibility
    logits = torch.sum(logits, 1)

    return logits


class Class_head_MetaOpt(nn.Module):
    def __init__(self, way,shot,query,enable_scale=True):
        super(Class_head_MetaOpt, self).__init__()
        self.head = MetaOptNetHead_SVM_CS
        # Add a learnable scale
        self.enable_scale = enable_scale
        self.scale = nn.Parameter(torch.FloatTensor([1.0]))

        self.n_way=way
        self.n_shot=shot
        self.q=query



    def get_out(self, logit_query, labels_query, n_way):
        smoothed_one_hot = one_hot(labels_query.reshape(-1), n_way)
        # smoothed_one_hot = smoothed_one_hot * (1 - opt.eps) + (1 - smoothed_one_hot) * opt.eps / (opt.train_way - 1)
        log_prb = F.log_softmax(logit_query.reshape(-1, n_way), dim=1)
        loss = -(smoothed_one_hot * log_prb).sum(dim=1)
        loss = loss.mean()
        return log_prb, loss
    # def forward(self, query, support, support_labels, query_labels, n_way, n_shot):
    def forward(self, feat,label):
        n_way=self.n_way
        n_shot=self.n_shot

        support_labels=label[0].repeat_interleave(n_shot).unsqueeze(0).cuda()
        query_labels=label[1].unsqueeze(0).cuda()

        support=feat[:self.n_way*self.n_shot]
        support=support.unsqueeze(0)
        query=feat[self.n_way*self.n_shot:].unsqueeze(0)
        
        

        if self.enable_scale:
            logit_query = self.scale * self.head(query, support, support_labels, n_way, n_shot)
            pred, loss = self.get_out(logit_query, query_labels, n_way)
        else:
            logit_query = self.head(query, support, support_labels, n_way, n_shot)
            pred, loss = self.get_out(logit_query, query_labels, n_way)
        return pred, loss


if __name__=='__main__':
    device='cuda'

    sample_inpt=torch.randn((20,512)).to(device)
    net=Class_head_MetaOpt(way=5,shot=1,query=3).to(device)
    s_label=torch.arange(5).to(device)
    q_label=torch.arange(5).repeat_interleave(3).to(device)
    pred,loss= net(sample_inpt,[s_label,q_label])


    # === get input ====
    n_way=5
    n_shot=1
    n_query=3
    device='cuda'
    bs=2
    MetaOp=Class_head_MetaOpt(n_way,n_shot,n_query).to(device)


    query=torch.randn((bs,n_way*n_query,512)).to(device)
    support=torch.randn((bs,n_way*n_shot,512)).to(device)
    query_label=torch.arange(n_way).repeat_interleave(n_query).to(device)
    support_label=torch.arange(n_way).repeat_interleave(n_shot).to(device)

    query_label=query_label.unsqueeze(0).repeat(bs,1)
    support_label=support_label.unsqueeze(0).repeat(bs,1)



    # ==================

    out=MetaOp(query,support,support_label,query_label,n_way,n_shot)
    a=1