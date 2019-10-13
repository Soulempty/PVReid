import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, Bottleneck

#without input feature:out.dim=input.dim
class PPM(nn.Module):
    def __init__(self, in_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, in_dim//len(bins), kernel_size=1, bias=False),
                nn.BatchNorm2d(in_dim//len(bins)),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)
        self.merge = nn.Sequential(nn.Conv2d(in_dim,in_dim,kernel_size=1,bias=False),nn.BatchNorm2d(in_dim))

    def forward(self, x):
        x_size = x.size()
        out = []
        for f in self.features:
            out.append(F.interpolate(f(x), (18,18), mode='bilinear', align_corners=True))
        return self.merge(torch.cat(out, 1))
    
class REFC(nn.Module):
    def __init__(self,feat_in,feat_out,num_classes):
        super(REFC, self).__init__()
        
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


class Sparker(nn.Module):
    def __init__(self, num_classes,bins=[1,2,3,6]):
        super(Sparker, self).__init__()
        resnet = resnet50(pretrained=True)
        self.layer1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
        ) 
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        self.layer4.load_state_dict(resnet.layer4.state_dict())

        self.maxp1 = nn.AdaptiveMaxPool2d((2,1))  # 2*1
        self.maxp2 = nn.AdaptiveMaxPool2d((1,2))  # 2*1 
        self.maxpool = nn.AdaptiveMaxPool2d(1)  # 2*1

        self.refc1 = REFC(2048, 512, num_classes)
        self.refc2 = REFC(512, 256, num_classes)
        self.refc3 = REFC(512, 256, num_classes)
        self.refc4 = REFC(1024, 256, num_classes)
        self.refc5 = REFC(1024, 256, num_classes)
        self.refc6 = REFC(2048, 256, num_classes)
        self.refc7 = REFC(2048, 256, num_classes)
        self.refc8 = REFC(2048, 256, num_classes)
        self.refc9 = REFC(2048, 256, num_classes)

    def forward(self, x):
        x = self.layer1(x)
        feature2 = self.layer2(x)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        
        pool1 = self.maxp1(feature2)
        pool2 = self.maxp1(feature3)
        pool21 = self.maxp1(feature4)
        pool12 = self.maxp2(feature4)
        pool = self.maxpool(feature4)

        feat1 = pool
        feat2,feat3 = pool1.chunk(2,dim=2)
        feat4,feat5 = pool2.chunk(2,dim=2)
        feat6,feat7 = pool21.chunk(2,dim=2)
        feat8,feat9 = pool12.chunk(2,dim=3)

        ft1, bn1, fc1 = self.refc1(feat1)
        ft2, bn2, fc2 = self.refc2(feat2)
        ft3, bn3, fc3 = self.refc3(feat3)
        ft4, bn4, fc4 = self.refc4(feat4)
        ft5, bn5, fc5 = self.refc5(feat5)
        ft6, bn6, fc6 = self.refc6(feat6)
        ft7, bn7, fc7 = self.refc7(feat7)
        ft8, bn8, fc8 = self.refc8(feat8)
        ft9, bn9, fc9 = self.refc9(feat9)

        predict = torch.cat([bn1, bn2, bn3, bn4, bn5, bn6, bn7, bn8, bn9], dim=1)

        return (ft1, ft2, ft3, ft4, ft5, ft6, ft7, ft8, ft9), (fc1, fc2, fc3, fc4, fc5, fc6, fc7, fc8, fc9), predict

