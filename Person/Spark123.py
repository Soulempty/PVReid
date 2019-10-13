import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck

class PoolBlock(nn.Module):
    def __init__(self, size=(),stride=2,dim=2):
        super().__init__()
        s = max(size)
        self.maxpool = nn.MaxPool2d(kernel_size=size,stride=stride)
        self.avgpool = nn.AvgPool2d(kernel_size=size,stride=stride)
        
        self.act = nn.Sigmoid()
        self.d = dim
        #self.act = nn.Softmax(dim=self.d)

    def forward(self, x):
        maxpool = self.maxpool(x)
        avgpool = self.avgpool(x)
        act =self.act(avgpool)
        
        f = act*maxpool
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
    
    
    
    
class Spark123(nn.Module):
    def __init__(self,num_classes):
        super(Spark123, self).__init__()

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
        
        #self.PB11 = PoolBlock(size=(3,6))#8*4
        #self.PB21 = PoolBlock(size=(6,3))#4*8
        
        self.PB1 = PoolBlock(size=(8,8),stride=8)#3*1
        self.PB2 = PoolBlock(size=(4,4),stride=4)#6*2      
        self.PB3 = PoolBlock(size=(8,4),stride=2)#9*3
        
        self.avgpool = nn.AvgPool2d(kernel_size=(24,8))
        self.maxpool = nn.MaxPool2d(kernel_size=(24,8))
        
        self.p3_1 = ReductionFc(2048,256,num_classes)
        self.p3_2 = ReductionFc(2048,256,num_classes)
        self.p3_3 = ReductionFc(2048,256,num_classes)
        self.p1_1 = ReductionFc(2048,256,num_classes)
        self.p2_1 = ReductionFc(2048,256,num_classes)
        self.p2_2 = ReductionFc(2048,256,num_classes)
 
        self.p_g1 = ReductionFc(638,256,num_classes)
        self.p_g2 = ReductionFc(638,256,num_classes)
        self.p_g3 = ReductionFc(638,256,num_classes)
        self.p_g4 = ReductionFc(638,256,num_classes)

    def forward(self, x):
        x = self.backbone(x)
        p1 = self.p1(x)

        p11 = self.PB1(p1)#16*2048*1*3
        p22 = self.PB2(p1)#16*2048*3*1
        p33 = self.PB3(p1)#16*2048*1*2
        
        p_g =self.avgpool(p1)+self.maxpool(p1)
        p_g1 = p_g[:, :638, :, :]
        p_g2 = p_g[:, 470:1108, :, :]
        p_g3 = p_g[:, 940:1578, :, :]
        p_g4 = p_g[:, 1410:2048, :, :]
        
        p3_f1,p3_f2,p3_f3 = p33.chunk(3,3)
        p2_f1,p2_f2 = p22.chunk(2,3) 
        p1_f1 = p11
        
        p3_r1, p3_fc1 = self.p3_1(p3_f1)
        p3_r2, p3_fc2 = self.p3_2(p3_f2)
        p3_r3, p3_fc3 = self.p3_3(p3_f3)
        p2_r1, p2_fc1 = self.p2_1(p2_f1)
        p2_r2, p2_fc2 = self.p2_2(p2_f2)
        p1_r1, p1_fc1 = self.p1_1(p1_f1)
   
        pg_r1, pg_fc1 = self.p_g1(p_g1)
        pg_r2, pg_fc2 = self.p_g2(p_g2)
        pg_r3, pg_fc3 = self.p_g3(p_g3)
        pg_r4, pg_fc4 = self.p_g4(p_g4)

        p1_at = p1_r1 #256
        p2_at = torch.cat([p2_r1,p2_r2],dim=1)#256*2=512
        p3_at = torch.cat([p3_r1,p3_r2,p3_r3],dim=1)#256*3=768
        
        pg_at = torch.cat([pg_r1,pg_r2,pg_r3,pg_r4],dim=1)#256*4=1024

        predict = torch.cat([p1_at, p2_at, p3_at, pg_at], dim=1)

        return (p1_at, p2_at, p3_at, pg_at), (p2_fc1, p2_fc2, p1_fc1, p3_fc1, p3_fc2, p3_fc3, pg_fc1, pg_fc2, pg_fc3, pg_fc4), predict

