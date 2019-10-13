import os
import re
import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
import torch
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from opt import opt
from datetime import datetime
from data import Data
#Vehicel Re-identification
from Vehicle.PowerNet import PowerNet
from Vehicle.SparkAct import SparkAct
from Vehicle.Spark1 import Spark1
from Vehicle.Spark2 import Spark2  
from Vehicle.Fighter import Fighter
from Vehicle.Spark3 import Spark3
from Vehicle.Spark_layer import Sparker
from Vehicle.baseline import Baseline
#Person Re-identification
from Person.SparkCN import Spark_CN
from Person.FlyDream import FlyDream
from Person.MGN import MGN
from Person.FlyDragon import FlyDragon
from Person.SparkPCA import SparkPCA
from Person.SparkPower import SparkPower
from Person.SparkPower4 import SparkPower4
from Person.Spark123 import Spark123

from loss import Loss
from utils.get_optimizer import get_optimizer
from utils.extract_feature import extract_feature
from utils.metrics import euclidean_dist,re_rank,getRank

os.environ['CUDA_VISIBLE_DEVICES'] = opt.device
Model = {'powernet':PowerNet,'sp1':Spark1,'sp2':Spark2,'sp3':Spark3,'spa':SparkAct,'base':Baseline,'fighter':Fighter,'spk':Sparker,
       'spcn':Spark_CN,'flydream':FlyDream,'flydragon':FlyDragon,'spca':SparkPCA,'spkp':SparkPower,'spk123':Spark123,'spkp4':SparkPower4}
class DoTraing():
    def __init__(self, model, loss, data,start_epoch=-1):
        self.train_loader = data.train_loader
        self.test_loader = data.test_loader
        self.query_loader = data.query_loader
        self.testset = data.testset
        self.queryset = data.queryset
        self.model = model.cuda()
        self.loss = loss
        self.flag=True
        self.start_epoch = start_epoch
        self.optimizer = get_optimizer(model)
        self.scheduler = lr_scheduler.MultiStepLR(self.optimizer, milestones=opt.lr_scheduler, gamma=0.1)

    def train(self):
        if self.start_epoch!=-1 and self.flag:
            self.flag=False
            for i in range(start_epoch):
                self.scheduler.step()
        else:
            self.scheduler.step()

        self.model.train()
        for batch, (inputs, labels) in enumerate(self.train_loader):
            inputs = inputs.cuda()
            labels = labels.cuda()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.loss(outputs, labels)
            loss.backward()
            self.optimizer.step()
        print('\nBatch:',batch)

    def evaluate(self):
        
        self.model.eval()  
        print('extract features, this may take a few minutes')
        qf = extract_feature(self.model, tqdm(self.query_loader)).numpy()
        gf = extract_feature(self.model, tqdm(self.test_loader)).numpy()

        def rank(dist):
            r = cmc(dist, self.queryset.pids,self.testset.pids,self.queryset.camids,self.testset.camids)
            m_ap = mean_ap(dist, self.queryset.pids,self.testset.pids,self.queryset.camids,self.testset.camids)

            return r, m_ap
        #########################no re rank##########################
        dist = cdist(qf, gf)
        r, m_ap = rank(dist)
        print('[Without Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))
        
        #########################   re rank##########################
        q_g_dist = np.dot(qf, np.transpose(gf))
        q_q_dist = np.dot(qf, np.transpose(qf))
        g_g_dist = np.dot(gf, np.transpose(gf))
        dist = re_ranking(q_g_dist, q_q_dist, g_g_dist)
        r, m_ap = rank(dist)
        print('[With    Re-Ranking] mAP: {:.4f} rank1: {:.4f} rank3: {:.4f} rank5: {:.4f} rank10: {:.4f}'
              .format(m_ap, r[0], r[2], r[4], r[9]))

        

def Main(data, model, loss):
    print("Class:",data.nums)
    if opt.ngpu>1:
        model=nn.DataParallel(model)
    start_epoch=-1
    if opt.resume and os.path.exists(opt.weight):
        start_epoch=int(re.sub("\D", "",opt.weight.split("/")[-1]))
        model.load_state_dict(torch.load(opt.weight))
    main = DoTraing(model, loss, data, start_epoch)
    if opt.mode == 'train':
        modelName = str(type(model)).split('.')[-1][:-2]
        print(modelName,"Training")
        for epoch in range(1+start_epoch, opt.epoch + 1):
            st = datetime.now()
            print('\nepoch', epoch)
            main.train()
            end = 1.0*(datetime.now()-st).seconds
            print('\nTime used:',round(end/60,2))
            if epoch % 10 == 0:
                print('\nstart evaluate')
                main.evaluate()
                if epoch >100 and epoch%20==0: 
                    savePath = 'Pretrained/'+modelName
                    os.makedirs(savePath, exist_ok=True)
                    torch.save(model.state_dict(), (savePath+'/model_{}.pt'.format(epoch)))
    if opt.mode == 'evaluate':
        print('start evaluate')
        model.load_state_dict(torch.load(opt.weight))
        main.evaluate()


if __name__ == '__main__':
    dtype = opt.dtype
    if dtype == 'person':
        size = (384,128)
    else:
        size = (288,288)
    data = Data(data=opt.data_name,size=size)
    model = Model[opt.name](data.nums)
    cudnn.benchmark = True
    loss = Loss()
    Main(data, model, loss)
