import os
import re
import numpy as np
from scipy.spatial.distance import cdist
from tqdm import tqdm
import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
from torch import nn
import torch
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from opt import opt
from data import Data
from PoolModel.psp_new import Res2Net
from PoolModel.psp_4 import Spark_PCA
from models.vehicle import VeRi
from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract import extract_feature
from utils.metrics import mean_ap, cmc, re_ranking

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

class Main():
    def __init__(self, model, loss, data,start_epoch=-1):
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset

        self.model = model.cuda()
        
    def evaluate(self):

        self.model.eval()
        txt = open(opt.save_path,"w")
        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()
        def rank(dist):
            r = cmc(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras,
                    separate_camera_set=False,
                    single_gallery_shot=False,
                    first_match_break=True)
            m_ap = mean_ap(dist, self.queryset.ids, self.testset.ids, self.queryset.cameras, self.testset.cameras)

            return r, m_ap

        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        distmat = re_ranking(q_g_dist, q_q_dist, g_g_dist)

        r, m_ap = rank(distmat)

        print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))
        #########################no re rank##########################
        dist = cdist(qf, gf)

        r, m_ap = rank(dist)

        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))
        f = self.result(distmat,txt)
    def result(self,distmat,txt):
        m, n = distmat.shape
        q_n= [f.split("/")[-1].split(".")[0] for f in self.queryset.imgs]
        g_n = [f.split("/")[-1].split(".")[0] for f in self.testset.imgs]
        indices = np.argsort(distmat, axis=1)
        top20=indices[:,:20]
        print('*******begin generate the top20 result***********')
        for i in range(len(top20)):
            name=q_n[i]+" "
            for s in range(20):
                name+=g_n[int(top20[i,s])]+' '
            name+='\n'
            txt.write(name)
        print('#########generate over#############')
        return True
if __name__ == '__main__':

    data = Data(data="vehicleid")
    model = Res2Net(num_classes=4481)
    cudnn.benchmark = True
    loss = Loss()
    start_epoch=-1
    main = Main(model, loss, data,start_epoch)
    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()
