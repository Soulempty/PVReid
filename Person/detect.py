import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision.models.resnet import resnet50, Bottleneck

class ReductionFc(nn.Module):
    def __init__(self,feat_in,feat_out,num_classes):
        super(ReductionFc, self).__init__()
        
        self.conv1 = nn.Conv2d(feat_in, feat_out, 1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_in')
        self.bn1 = nn.BatchNorm1d(feat_out)
        nn.init.normal_(self.bn1.weight, mean=1., std=0.02)
        nn.init.constant_(self.bn1.bias, 0.)
        self.fc = nn.Linear(feat_out, num_classes,bias=False)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')

    def forward(self,x):
        reduce = self.conv1(x).view(x.size(0),-1)
        bn = self.bn1(reduce)
        fc = self.fc(bn)
        
        return reduce,bn,fc
    
    
    
    
class Attention(nn.Module):
    def __init__(self,num_classes,num_local=4):
        super(Attention, self).__init__()
        self.num_local = num_local
        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )
        self.conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        self.conv5.load_state_dict(resnet.layer4.state_dict())        
        self.local_list = nn.ModuleList()
        for i in range(num_local):
            self.local_list.append(nn.Sequential(nn.Conv2d(2048,256,1,bias=False),
                                     nn.BatchNorm2d(256),
                                     nn.ReLU(inplace=True)))#relu
        self.local_fcs = nn.ModuleList()
        for _ in range(num_local):
            fc = nn.Linear(256,num_classes,bias=False)
            nn.init.normal_(fc.weight,std=0.001)
            self.local_fcs.append(fc) 
        self.avgpool = nn.AvgPool2d(kernel_size=(24,24))
        self.maxpool = nn.MaxPool2d(kernel_size=(24,24))    

    def forward(self, x,box1,box2,box3,box4):
        x = self.backbone(x)
        p1 = self.conv5(x)#16*2048*24*24
        feat_list = []
        logit_list = []
        N, C, H, W = p1.shape
        feat_window = Variable(torch.zeros((N,C,1,1)).cuda())
        feat_light = Variable(torch.zeros((N,C,1,1)).cuda())
        feat_brand = Variable(torch.zeros((N,C,1,1)).cuda())
        for i in range(N):
            light0 = p1[i,:,box2[i][1]:box2[i][3],box2[i][0]:box2[i][2]]
            light1 = p1[i,:,box3[i][1]:box3[i][3],box3[i][0]:box3[i][2]]
            if not (len(light0.size()[1:]) ==2 and len(light1.size()[1:]) ==2):
                print('light size')
                light0 = p1[i, :, 16:24, 0:8]
                light1 = p1[i, :, 16:24,16:24]
            light0 = F.avg_pool2d(light0, light0.size()[1:])
            light1 = F.avg_pool2d(light1, light1.size()[1:])
            feat_light[i] = (light0+light1)/2

            brand = p1[i,:,box4[i][1]:box4[i][3],box4[i][0]:box4[i][2]]
            feat_brand[i] = F.avg_pool2d(brand, brand.size()[1:])
            try:
                window = p1[i,:,box1[i][1]:box1[i][3],box1[i][0]:box1[i][2]]
            except:
                print ('window except')
            feat_window[i] = F.avg_pool2d(window,window.size()[1:])
        g_feat = self.avgpool(p1)+self.maxpool(p1)
        for i,local_feat in enumerate([feat_light,feat_window,feat_brand,g_feat]):
            try:
                local_feat = self.local_list[i](local_feat)
                local_feat = local_feat.view(local_feat.size(0), -1)
                feat_list.append(local_feat)
                logit_list.append(self.local_fcs[i](local_feat))
            except:
                print ('conv fc')

        return feat_list, logit_list, feat_list[-1]
