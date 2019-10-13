import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet50, Bottleneck

# Apply PPM with ADD operation
class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = []
        for bin in bins:
            self.features.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ))
        self.features = nn.ModuleList(self.features)

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)
    
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


class SparkAct(nn.Module):
    def __init__(self, num_classes,bins=[1,2,3,6]):
        super(SparkAct, self).__init__()
        resnet = resnet50(pretrained=True)
        layer4 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        layer4.load_state_dict(resnet.layer4.state_dict())
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            layer4
        )
        self.branch1 = nn.Conv2d(2048, 1024, 1, bias=False)
        self.branch2 = nn.Conv2d(2048, 1024, 1, bias=False)
        self.branch3 = nn.Conv2d(2048, 1024, 1, bias=False)

        
        self.maxp1 = nn.MaxPool2d(kernel_size=(9, 9))  
        self.maxp2 = nn.MaxPool2d(kernel_size=(3, 6))  
        self.maxp3 = nn.MaxPool2d(kernel_size=(9, 3)) 
        
        self.avgp1 = nn.AvgPool2d(kernel_size=(9, 9))  
        self.avgp2 = nn.AvgPool2d(kernel_size=(3, 6))  
        self.avgp3 = nn.AvgPool2d(kernel_size=(9, 3)) 
        self.maxpool = nn.MaxPool2d(kernel_size=(18, 18))  # 2*1
        self.avgpool = nn.AvgPool2d(kernel_size=(2, 2))
        self.sig = nn.Sigmoid()
        self.vsof = nn.Softmax(dim=2)
        self.hsof = nn.Softmax(dim=3)
        # total 3+2+3+4 branch for classificaiton
        self.refc1 = REFC(1024, 256, num_classes)
        self.refc2 = REFC(1024, 256, num_classes)
        self.refc3 = REFC(1024, 256, num_classes)
        self.refc4 = REFC(1024, 256, num_classes)
        self.refc5 = REFC(1024, 256, num_classes)
        self.refc6 = REFC(1024, 256, num_classes)
        self.refc7 = REFC(2048, 256, num_classes)


    def forward(self, x):
        feature = self.backbone(x)
        feat7 = self.maxpool(feature)
        branch1 = self.branch1(feature)
        branch2 = self.branch2(feature)
        branch3 = self.branch3(feature)
        
        feat1 = self.avgpool(self.maxp1(branch1)*self.sig(self.avgp1(branch1)))
        pool2 = (self.maxp2(branch2)*self.vsof(self.avgp2(branch2))).sum(2,keepdim=True)
        pool3 = (self.maxp3(branch3)*self.hsof(self.avgp3(branch3))).sum(3,keepdim=True)
        feat2,feat3,feat4 = pool2.chunk(3,dim=3)
        feat5,feat6 = pool3.chunk(2,dim=2)
        
        ft1, bn1, fc1 = self.refc1(feat1)
        ft2, bn2, fc2 = self.refc2(feat2)
        ft3, bn3, fc3 = self.refc3(feat3)
        ft4, bn4, fc4 = self.refc4(feat4)
        ft5, bn5, fc5 = self.refc5(feat5)
        ft6, bn6, fc6 = self.refc6(feat6)
        ft7, bn7, fc7 = self.refc7(feat7)

        predict = torch.cat([bn1, bn2, bn3, bn4, bn5, bn6, bn7], dim=1)

        return (ft1, ft2, ft3, ft4, ft5, ft6, ft7), (fc1, fc2, fc3, fc4, fc5, fc6, fc7), predict

