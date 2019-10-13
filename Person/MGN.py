import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck


class ReductionFc(nn.Module):
    def __init__(self,feat_in,feat_out,num_classes,part=True):
        super(ReductionFc, self).__init__()
        self.part = part
        self.reduction = nn.Sequential(nn.Conv2d(feat_in, feat_out, 1, bias=False), nn.BatchNorm2d(feat_out), nn.ReLU())
        self._init_reduction(self.reduction)
        self.fc1 = nn.Linear(feat_out, num_classes,bias=False)
        self._init_fc(self.fc1)
        self.fc2 = nn.Linear(feat_in, num_classes,bias=False)
        self._init_fc(self.fc2)
    
    @staticmethod
    def _init_reduction(reduction):
        # conv
        nn.init.kaiming_normal_(reduction[0].weight, mode='fan_in')
        # nn.init.constant_(reduction[0].bias, 0.)

        # bn
        nn.init.normal_(reduction[1].weight, mean=1., std=0.02)
        nn.init.constant_(reduction[1].bias, 0.)

    @staticmethod
    def _init_fc(fc):
        nn.init.kaiming_normal_(fc.weight, mode='fan_out')
        # nn.init.normal_(fc.weight, std=0.001)
        #nn.init.constant_(fc.bias, 0.)
    def forward(self,x):
        reduce = self.reduction(x).view(x.size(0),-1)
        if self.part:
            fc_256 = self.fc1(reduce)
            return reduce, fc_256
        else:
            x1 = x.view(x.size(0),-1)
            fc_2048 = self.fc2(x1)
            return reduce, fc_2048
        
    
class MGN(nn.Module):
    def __init__(self,num_classes):
        super(MGN, self).__init__()

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

        res_g_conv5 = resnet.layer4

        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_zg_p1 = nn.MaxPool2d(kernel_size=(12, 4))
        self.maxpool_zg_p2 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zg_p3 = nn.MaxPool2d(kernel_size=(24, 8))
        self.maxpool_zp2 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_zp3 = nn.MaxPool2d(kernel_size=(8, 8))

        self.fc_g1 = ReductionFc(2048,256,num_classes,part=False)
        self.fc_g2 = ReductionFc(2048,256,num_classes,part=False)
        self.fc_g3 = ReductionFc(2048,256,num_classes,part=False)
        
        self.fc_p1_1 = ReductionFc(2048,256,num_classes)
        self.fc_p1_2 = ReductionFc(2048,256,num_classes)
        self.fc_p2_1 = ReductionFc(2048,256,num_classes)
        self.fc_p2_2 = ReductionFc(2048,256,num_classes)
        self.fc_p2_3 = ReductionFc(2048,256,num_classes)

    def forward(self, x):
        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        zg_p1 = self.maxpool_zg_p1(p1)#three global 
        zg_p2 = self.maxpool_zg_p2(p2)
        zg_p3 = self.maxpool_zg_p3(p3)
        
        zp2 = self.maxpool_zp2(p2)#five branch
        z0_p2 = zp2[:, :, 0:1, :]
        z1_p2 = zp2[:, :, 1:2, :]

        zp3 = self.maxpool_zp3(p3)
        z0_p3 = zp3[:, :, 0:1, :]
        z1_p3 = zp3[:, :, 1:2, :]
        z2_p3 = zp3[:, :, 2:3, :]
        #
        fg_p1, f0_p1 = self.fc_g1(zg_p1)#reduce triplet feature, fc_id_2048
        fg_p2, f0_p2 = self.fc_g2(zg_p2)
        fg_p3, f0_p3 = self.fc_g3(zg_p3)
        
        ft1_p1, f1_p1 = self.fc_p1_1(z0_p2)#reduce feature, fc_id_256
        ft2_p1, f2_p1 = self.fc_p1_2(z1_p2)
        
        ft1_p2, f1_p2 = self.fc_p2_1(z0_p3)
        ft2_p2, f2_p2 = self.fc_p2_2(z1_p3)
        ft3_p2, f3_p2 = self.fc_p2_3(z2_p3)
        

        predict = torch.cat([fg_p1, fg_p2, fg_p3, ft1_p1, ft2_p1, ft1_p2, ft2_p2, ft3_p2], dim=1)

        return (fg_p1,fg_p2,fg_p3), (f0_p1,f0_p2,f0_p3,f1_p1,f2_p1,f1_p2,f2_p2,f3_p2), predict
