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


class Baseline(nn.Module):
    def __init__(self, num_classes,bins=[1,2,3,6]):
        super(Baseline, self).__init__()
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
        
        #self.maxpool = nn.MaxPool2d(kernel_size=(18, 18)) 
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.bn = nn.BatchNorm1d(2048)
        nn.init.normal_(self.bn.weight, mean=1., std=0.02)
        nn.init.constant_(self.bn.bias, 0.)
        self.fc = nn.Linear(2048, num_classes,bias=False)
        nn.init.kaiming_normal_(self.fc.weight, mode='fan_out')


    def forward(self, x):
        feature = self.backbone(x)
        feat = self.maxpool(feature)
        feat = feat.view(feat.shape[0], -1)
        bn = self.bn(feat)
        fc = self.fc(bn)
        return feat, fc, bn

