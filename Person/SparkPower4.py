import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck

class PoolBlock(nn.Module):
    def __init__(self, size=(),stride=(),dim=3):
        super().__init__()
        s = max(size)
        self.maxpool = nn.MaxPool2d(kernel_size=size,stride=stride)
        self.avgpool = nn.AvgPool2d(kernel_size=size,stride=stride)
        self.gap = nn.AvgPool2d(kernel_size=3,stride=3)
        self.act = nn.Sigmoid()
        self.d = dim
        #self.act = nn.Softmax(dim=self.d)

    def forward(self, x):
        maxpool = self.maxpool(x)
        avgpool = self.avgpool(x)
        act =self.act(avgpool)
        
        f = act*maxpool
        if self.d==23:
            fuse_f = self.gap(f)
        else:
            fuse_f = f.mean(self.d,keepdim=True)
        #fuse_f = f.sum(self.d,keepdim=True)
        return fuse_f
    
class ReductionFc(nn.Module):
    def __init__(self,feat_in,feat_out,num_classes):
        super(ReductionFc, self).__init__()
        
        self.reduction = nn.Sequential(nn.Conv2d(feat_in, feat_out, 1, bias=False), nn.BatchNorm2d(feat_out))#, nn.ReLU()
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
    
    
    
    
class SparkPower4(nn.Module):
    def __init__(self,num_classes):
        super(SparkPower4, self).__init__()

        resnet = resnet50(pretrained=True)
        #modify the first conv
        #resnet.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
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
        
        self.PB1 = PoolBlock(size=(12,4),stride=(12,2))#2*3->2
        self.PB2 = PoolBlock(size=(8,5),stride=(8,3))#3*2->3
        self.PB3 = PoolBlock(size=(6,8),stride=(6,8))#4*1->4
        self.PB4 = PoolBlock(size=(12,4),stride=(6,2),dim=23)#3*3->1
        
        self.avgpool = nn.AvgPool2d(kernel_size=(24,8))
        self.maxpool = nn.MaxPool2d(kernel_size=(24,8))
        
        self.p1_1 = ReductionFc(2048,256,num_classes)
        self.p1_2 = ReductionFc(2048,256,num_classes)
        self.p2_1 = ReductionFc(2048,256,num_classes)
        self.p2_2 = ReductionFc(2048,256,num_classes)
        self.p2_3 = ReductionFc(2048,256,num_classes)
        self.p3_1 = ReductionFc(2048,256,num_classes)
        self.p3_2 = ReductionFc(2048,256,num_classes)
        self.p3_3 = ReductionFc(2048,256,num_classes)
        self.p3_4 = ReductionFc(2048,256,num_classes)
        self.p4_1 = ReductionFc(2048,256,num_classes)
        
        self.p_g1 = ReductionFc(638,256,num_classes)
        self.p_g2 = ReductionFc(638,256,num_classes)
        self.p_g3 = ReductionFc(638,256,num_classes)
        self.p_g4 = ReductionFc(638,256,num_classes)

    def forward(self, x):
        x = self.backbone(x)
        p1 = self.p1(x)

        p11 = self.PB1(p1)#16*2048*2*1
        p22 = self.PB2(p1)#16*2048*3*1
        p33 = self.PB3(p1)#16*2048*4*1
        p44 = self.PB4(p1)#16*2048*1*1
        
        p_g =self.avgpool(p1)+self.maxpool(p1)
        p_g1 = p_g[:, :638, :, :]
        p_g2 = p_g[:, 470:1108, :, :]
        p_g3 = p_g[:, 940:1578, :, :]
        p_g4 = p_g[:, 1410:2048, :, :]
        
        p11_f1,p11_f2 = p11.chunk(2,2)
        p22_f1,p22_f2,p22_f3 = p22.chunk(3,2) 
        p33_f1,p33_f2,p33_f3, p33_f4 = p33.chunk(4,2)
        p44_f1 = p44 

        
        p11_r1, p11_fc1 = self.p1_1(p11_f1)
        p11_r2, p11_fc2 = self.p1_2(p11_f2)
        
        p22_r1, p22_fc1 = self.p2_1(p22_f1)
        p22_r2, p22_fc2 = self.p2_2(p22_f2)
        p22_r3, p22_fc3 = self.p2_3(p22_f3)

        p33_r1, p33_fc1 = self.p3_1(p33_f1)
        p33_r2, p33_fc2 = self.p3_2(p33_f2)
        p33_r3, p33_fc3 = self.p3_3(p33_f3)
        p33_r4, p33_fc4 = self.p3_4(p33_f4)
        
        p44_r1, p44_fc1 = self.p4_1(p44_f1)
        
        pg_r1, pg_fc1 = self.p_g1(p_g1)
        pg_r2, pg_fc2 = self.p_g2(p_g2)
        pg_r3, pg_fc3 = self.p_g3(p_g3)
        pg_r4, pg_fc4 = self.p_g4(p_g4)

        p11_at = torch.cat([p11_r1,p11_r2],dim=1)#256*4=1024
        p22_at = torch.cat([p22_r1,p22_r2,p22_r3],dim=1)#256*4=1024
        p33_at = torch.cat([p33_r1,p33_r2,p33_r3, p33_r4],dim=1)#256*2=512
        p44_at = p44_r1
        pg_at = torch.cat([pg_r1,pg_r2,pg_r3,pg_r4],dim=1)#256*4=1024

        predict = torch.cat([p11_at, p22_at, p33_at, p44_at, pg_at], dim=1)

        return (p11_at, p22_at, p33_at, p44_at, pg_at), (p22_fc1, p22_fc2, p22_fc3, p11_fc1, p11_fc2, p33_fc1, p33_fc2, p33_fc3, p33_fc4, p44_fc1, pg_fc1, pg_fc2, pg_fc3, pg_fc4), predict
