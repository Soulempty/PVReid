from __future__ import print_function, absolute_import
import time
import os
import torch
import numpy as np
from .re_ranking import re_ranking
from torch.autograd import Variable
from utils import to_numpy
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
def extract_feat(model, inputs):
    use_cuda=True
    hidden_dim=1024
    num_layers=1
    #h0 = 0.0 * torch.randn(num_layers,len(inputs), hidden_dim)
    if use_cuda:
        inputs = Variable(inputs.cuda())
        #h0=Variable(h0.cuda())
    else:
        inputs = Variable(inputs)
        #h0=Variable(h0)
    model.eval()
    tmp = model(inputs)
    outputs = tmp[9]
    outputs = outputs.data.cpu()
    return outputs

def extract_features(model, data_loader, print_freq=10):
    model.eval()  
    features = []
    for i, imgs in enumerate(data_loader):
        outputs = extract_feat(model, imgs)
        for output in outputs:
            features.append(output)
    return features


def pairwise_distance(query_features, gallery_features,dist='cosin'):

    x = torch.cat([query_features[i].unsqueeze(0) for i in range(len(query_features))], 0)

    y = torch.cat([gallery_features[i].unsqueeze(0) for i in range(len(gallery_features))], 0)
    
    m, n = x.size(0), y.size(0)
    #欧氏距离计算步骤
    if dist!='cosin':
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist = torch.pow(x, 2).sum(1).unsqueeze(1).expand(m, n) + \
           torch.pow(y, 2).sum(1).unsqueeze(1).expand(n, m).t()
        dist.addmm_(1, -2, x, y.t())#(x1-y1)**2+(x2-y2)**2==x1**2+x2**2+y1**2+y2**2-2*(x1*y1+x2*y2)
    else:
        x = x.view(m, -1)
        y = y.view(n, -1)
        dist=x.mm(y.t())#m*n (-1,1)
        #dist=0.5+dist*0.5
    dist = to_numpy(dist)
    print('dist shape:',dist.shape)
    return dist


def result(distmat,txt,dataset="aicity"):
    name={"veri":"VeRi","vihicleid":"VehicleID",'aicity':'AiCity'}
    dataName=name[dataset]
     
    #distmat = to_numpy(distmat)
    m, n = distmat.shape
    indices = np.argsort(distmat, axis=1)
    
    top100=indices[:,:100]
    print('*******begin generate the top100 result of ',dataName,'******')
    for i in range(len(top100)):
        name=''
        for s in range(100):
            name+=str(int(top100[i,s])+1)+' '
        name+='\n'
        txt.write(name)
    print('#########generate over#############')
    return True


class Evaluator1(object):
    def __init__(self, model):
        super(Evaluator1, self).__init__()
        self.model = model

    def evaluate(self, query_loader, gallery_loader,txt):
        txt=open(txt,'w')
        print('extracting query features\n')
        query_features = extract_features(self.model, query_loader)
        print('extracting gallery features\n')  
        gallery_features= extract_features(self.model, gallery_loader)
        
        qq_distmat = pairwise_distance(query_features, query_features)
        qg_dismat = pairwise_distance(query_features, gallery_features)
        gg_dismat = pairwise_distance(gallery_features, gallery_features)
        print('Reranking the result')
        re_rank = re_ranking(qg_dismat, qq_distmat, gg_dismat)
        return result(re_rank, txt)
