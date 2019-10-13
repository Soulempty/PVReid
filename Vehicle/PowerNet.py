import copy
import torch
import torch.nn as nn
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
    
    
    
    
class PowerNet(nn.Module):
    def __init__(self,num_classes):
        super(PowerNet, self).__init__()

        resnet = resnet50(pretrained=True)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3[0],
        )

        res_conv4 = nn.Sequential(*resnet.layer3[1:])
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))  
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))  

        self.maxpool_g1 = nn.MaxPool2d(kernel_size=(18, 18))#2*1
        self.maxpool_g2 = nn.MaxPool2d(kernel_size=(18, 18))#2*1
        
        self.maxpool_v = nn.MaxPool2d(kernel_size=(18, 9))#3
        self.maxpool_h = nn.MaxPool2d(kernel_size=(9, 18))#2

        #total 3+2+3+4 branch for classificaiton
        self.fc_g_1 = ReductionFc(1024,256,num_classes)
        self.fc_g_2 = ReductionFc(1024,256,num_classes)
        self.fc_g_3 = ReductionFc(1024,256,num_classes)
        self.fc_g_4 = ReductionFc(1024,256,num_classes)
        
        self.fc_h_1 = ReductionFc(2048,256,num_classes)
        self.fc_h_2 = ReductionFc(2048,256,num_classes)

        self.fc_v_1 = ReductionFc(2048,256,num_classes)
        self.fc_v_2 = ReductionFc(2048,256,num_classes)
        #       

    def forward(self, x):
        x = self.backbone(x)
        p1 = self.p1(x)
        p2 = self.p2(x)
        
        pg1 = self.maxpool_g1(p1)
        
        pg2 = self.maxpool_g2(p2)

        v_a = self.maxpool_v(p1)
        v_1 = v_a[:, :, :, 0:1]
        v_2 = v_a[:, :, :, 1:2]

        h_a = self.maxpool_h(p2)
        h_1 = h_a[:, :, 0:1, :]
        h_2 = h_a[:, :, 1:2, :]

        #
        g1, bn1, fc1 = self.fc_g_1(pg1[:,:1024,:,:])
        g2, bn2, fc2 = self.fc_g_2(pg2[:,:1024,:,:])
        g3, bn7, fc7 = self.fc_g_3(pg1[:,1024:2048,:,:])
        g4, bn8, fc8 = self.fc_g_4(pg2[:,1024:2048,:,:])
        h1, bn3, fc3 = self.fc_h_1(h_1)
        h2, bn4, fc4 = self.fc_h_2(h_2)

        v1, bn5, fc5 = self.fc_v_1(v_1)
        v2, bn6, fc6 = self.fc_v_2(v_2)

        at1 = torch.cat([h1, h2], dim=1)
        at2 = torch.cat([v1, v2], dim=1)
        predict = torch.cat([bn1, bn2, bn3, bn4, bn5, bn6, bn7,bn8], dim=1)

        return (at1, at2, g1, g2, g3, g4), (fc1, fc2, fc3, fc4, fc5, fc6,fc7,fc8), predict
