import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck

class ReductionFc(nn.Module):
    def __init__(self,feat_in,feat_out,num_classes):
        super(ReductionFc, self).__init__()
        
        self.reduction = nn.Sequential(nn.Conv2d(feat_in, feat_out, 1, bias=False), nn.BatchNorm2d(feat_out), nn.ReLU())
        self._init_reduction(self.reduction)
        self.fc = nn.Linear(feat_out, num_classes,bias=False)
        self._init_fc(self.fc)
    
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
        fc = self.fc(reduce)
        
        return reduce,fc
    
    
    
    
class FlyDragon(nn.Module):
    def __init__(self,num_classes):
        super(FlyDragon, self).__init__()

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
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))

        self.maxpool_g = nn.MaxPool2d(kernel_size=(24, 8))
        
        self.maxpool_p1 = nn.MaxPool2d(kernel_size=(12, 8))
        self.maxpool_p2 = nn.MaxPool2d(kernel_size=(8, 8))
        self.maxpool_p3 = nn.MaxPool2d(kernel_size=(6, 8))

        #total 3+2+3+4 branch for classificaiton
        self.fc_g1 = ReductionFc(2048,256,num_classes)
        self.fc_g2 = ReductionFc(2048,256,num_classes)
        self.fc_g3 = ReductionFc(2048,256,num_classes)

        self.fc_p1_1 = ReductionFc(2048,256,num_classes)
        self.fc_p1_2 = ReductionFc(2048,256,num_classes)
        
        self.fc_p2_1 = ReductionFc(2048,256,num_classes)
        self.fc_p2_2 = ReductionFc(2048,256,num_classes)
        self.fc_p2_3 = ReductionFc(2048,256,num_classes)
        
        self.fc_p3_1 = ReductionFc(2048,256,num_classes)
        self.fc_p3_2 = ReductionFc(2048,256,num_classes)
        self.fc_p3_3 = ReductionFc(2048,256,num_classes)
        self.fc_p3_4 = ReductionFc(2048,256,num_classes)
        #
        


    def forward(self, x):
        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)

        p1_g = self.maxpool_g(p1)#three global 
        p2_g = self.maxpool_g(p2)
        p3_g = self.maxpool_g(p3)

        p1_a = self.maxpool_p1(p1)#five branch
        p1_1 = p1_a [:, :, 0:1, :]
        p1_2 = p1_a [:, :, 1:2, :]

        p2_a = self.maxpool_p2(p2)
        p2_1 = p2_a[:, :, 0:1, :]
        p2_2 = p2_a[:, :, 1:2, :]
        p2_3 = p2_a[:, :, 2:3, :]
        
        p3_a = self.maxpool_p3(p3)
        p3_1 = p3_a[:, :, 0:1, :]
        p3_2 = p3_a[:, :, 1:2, :]
        p3_3 = p3_a[:, :, 2:3, :]
        p3_4 = p3_a[:, :, 3:4, :]
        
        #
        trip_g1, p1_g_fc = self.fc_g1(p1_g)
        trip_g2, p2_g_fc = self.fc_g2(p2_g)
        trip_g3, p3_g_fc = self.fc_g3(p3_g)
        
        p1_1_r, p1_1_fc = self.fc_p1_1(p1_1)
        p1_2_r, p1_2_fc = self.fc_p1_2(p1_2)
        
        p2_1_r, p2_1_fc = self.fc_p2_1(p2_1)
        p2_2_r, p2_2_fc = self.fc_p2_2(p2_2)
        p2_3_r, p2_3_fc = self.fc_p2_3(p2_3)
        
        p3_1_r, p3_1_fc = self.fc_p3_1(p3_1)
        p3_2_r, p3_2_fc = self.fc_p3_2(p3_2)
        p3_3_r, p3_3_fc = self.fc_p3_3(p3_3)
        p3_4_r, p3_4_fc = self.fc_p3_4(p3_4)
        
        trip_p1 = torch.cat([p1_1_r, p1_2_r],dim=1) 
        trip_p2 = torch.cat([p2_1_r, p2_2_r,p2_3_r],dim=1)
        trip_p3 = torch.cat([p3_1_r, p3_2_r,p3_3_r,p3_4_r],dim=1)

        predict = torch.cat([trip_p1, trip_p2, trip_p3, trip_g1, trip_g2, trip_g3], dim=1)

        return (trip_p1, trip_p2, trip_p3, trip_g1, trip_g2, trip_g3), (p1_g_fc, p2_g_fc, p3_g_fc, p1_1_fc, p1_2_fc, p2_1_fc, p2_2_fc, p2_3_fc, p3_1_fc, p3_2_fc, p3_3_fc, p3_4_fc), predict
