import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck

class ConvBNPReLU(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        args:
            nIn: number of input channels
            nOut: number of output channels
            kSize: kernel size
            stride: stride rate for down-sampling. Default is 1
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False)
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output

class BNPReLU(nn.Module):
    def __init__(self, nOut):
        """
        args:
           nOut: channels of output feature maps
        """
        super().__init__()
        self.bn = nn.BatchNorm2d(nOut, eps=1e-03)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: normalized and thresholded feature map
        """
        output = self.bn(input)
        output = self.act(output)
        return output
    
class ChannelWiseConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1):
        """
        Args:
            nIn: number of input channels
            nOut: number of output channels, default (nIn == nOut)
            kSize: kernel size
            stride: optional stride rate for down-sampling
        """
        super().__init__()
        padding = int((kSize - 1)/2)
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups=nIn, bias=False)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class ChannelWiseDilatedConv(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, default (nIn == nOut)
           kSize: kernel size
           stride: optional stride rate for down-sampling
           d: dilation rate
        """
        super().__init__()
        padding = int((kSize - 1)/2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), groups= nIn, bias=False, dilation=d)

    def forward(self, input):
        """
        args:
           input: input feature map
           return: transformed feature map
        """
        output = self.conv(input)
        return output

class FGlo(nn.Module):
    """
    the FGlo class is employed to refine the joint feature of both local feature and surrounding context.
    """
    def __init__(self, channel, reduction=16):
        super(FGlo, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(channel, channel // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(channel // reduction, channel),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class ContextGuidedBlock(nn.Module):
    def __init__(self, nIn, nOut, dilation_rate=1, reduction=16, add=True):
        """
        args:
           nIn: number of input channels
           nOut: number of output channels, 
           add: if true, residual learning
        """
        super().__init__()
        n= int(nOut/4)
        self.conv1x1 = ConvBNPReLU(nIn, n, 1, 1)  #1x1 Conv is employed to reduce the computation
        self.F_loc = ChannelWiseConv(n, n, 3, 1) # local feature
        self.F_sur1 = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate*2) # surrounding context
        self.F_sur2 = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate*3) # surrounding context
        self.F_sur3 = ChannelWiseDilatedConv(n, n, 3, 1, dilation_rate*4) # surrounding context
        self.bn_prelu = BNPReLU(nOut)
        self.add = add
        self.F_glo= FGlo(nOut, reduction)

    def forward(self, input):
        output = self.conv1x1(input)
        loc = self.F_loc(output)
        sur1 = self.F_sur1(output)
        sur2 = self.F_sur2(output)
        sur3 = self.F_sur3(output)
        
        joi_feat = torch.cat([loc, sur1,sur2,sur3], 1) 

        joi_feat = self.bn_prelu(joi_feat)

        output = self.F_glo(joi_feat)  #F_glo is employed to refine the joint feature
        # if residual version
        if self.add:
            output  = input + output
        return output

class PoolBlock(nn.Module):
    def __init__(self, size=2):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(output_size=(size, size))
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(size, size)) 

    def forward(self, x):
        maxpool = self.maxpool(x)
        avgpool = self.avgpool(x)
        fuse_f = avgpool+maxpool

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
    
    
    
    
class Spark_CN(nn.Module):
    def __init__(self,num_classes):
        super(Spark_CN, self).__init__()

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
        res_g_conv5 = resnet.layer4
        res_p_conv5 = nn.Sequential(
            Bottleneck(1024, 512, downsample=nn.Sequential(nn.Conv2d(1024, 2048, 1, bias=False), nn.BatchNorm2d(2048))),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512))
        res_p_conv5.load_state_dict(resnet.layer4.state_dict())

        self.p1 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_g_conv5))
        self.p2 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        self.p3 = nn.Sequential(copy.deepcopy(res_conv4), copy.deepcopy(res_p_conv5))
        
        self.CGBlock1 = ContextGuidedBlock(2048, 2048, dilation_rate=1, reduction=16, add=True)
        self.CGBlock2 = ContextGuidedBlock(2048, 2048, dilation_rate=2, reduction=16, add=True)
        self.CGBlock3 = ContextGuidedBlock(2048, 2048, dilation_rate=3, reduction=16, add=True)
        
        self.PB_g1 = PoolBlock(size=1)
        self.PB_g2 = PoolBlock(size=1)
        self.PB_g3 = PoolBlock(size=1)
        
        self.PB2 = PoolBlock(size=2)
        #self.PB3 = PoolBlock(size=3)
        
        self.p1_1 = ReductionFc(2048,256,num_classes)#
        
        self.p2_1 = ReductionFc(2048,256,num_classes)
        self.p2_2 = ReductionFc(2048,256,num_classes)
        self.p2_3 = ReductionFc(2048,256,num_classes)
        self.p2_4 = ReductionFc(2048,256,num_classes)
        self.p2_g = ReductionFc(2048,256,num_classes)#
        
        self.p3_1 = ReductionFc(638,256,num_classes)
        self.p3_2 = ReductionFc(638,256,num_classes)
        self.p3_3 = ReductionFc(638,256,num_classes)
        self.p3_4 = ReductionFc(638,256,num_classes)
        self.p3_g = ReductionFc(2048,256,num_classes)#
        

        #       

    def forward(self, x):
        x = self.backbone(x)
        p1 = self.p1(x)
        p2 = self.p2(x)
        p3 = self.p3(x)
        
        p1_f = self.CGBlock1(p1)
        p2_f = self.CGBlock2(p2)
        p3_f = self.CGBlock3(p3)
        #3 1024 global feature vector
        p1_1 = self.PB_g1(p1_f)#16*1024*1*1
        p2_1 = self.PB_g2(p2_f)#16*1024*1*1
        p3_1 = self.PB_g3(p3_f)#16*1024*1*1
        
        p3_f1 = p3_1[:, :638, :, :]
        p3_f2 = p3_1[:, 470:1108, :, :]
        p3_f3 = p3_1[:, 940:1578, :, :]
        p3_f4 = p3_1[:, 1410:2048, :, :]
        
        p2_4 = self.PB2(p2_f).view(p2_f.size(0),p2_f.size(1),-1)#16*1024*4
        p2_f1,p2_f2,p2_f3,p2_f4 = [f.unsqueeze(3) for f in p2_4.chunk(p2_4.size(2),2)]
        
        p1_rg, p1_g = self.p1_1(p1_1)
        
        p2_r1, p2_fc1 = self.p2_1(p2_f1)
        p2_r2, p2_fc2 = self.p2_2(p2_f2)
        p2_r3, p2_fc3 = self.p2_3(p2_f3)
        p2_r4, p2_fc4 = self.p2_4(p2_f4)
        p2_rg, p2_g = self.p2_g(p2_1)
        
        p3_r1, p3_fc1 = self.p3_1(p3_f1)
        p3_r2, p3_fc2 = self.p3_2(p3_f2)
        p3_r3, p3_fc3 = self.p3_3(p3_f3)
        p3_r4, p3_fc4 = self.p3_4(p3_f4)
        p3_rg, p3_g = self.p3_g(p3_1)
        
        p1_t = p1_rg#256
        p2_t = p2_rg#256
        p3_t = p3_rg#256
        p2_at = torch.cat([p2_r1,p2_r2,p2_r3,p2_r4],dim=1)#256*4=1024
        p3_at = torch.cat([p3_r1,p3_r2,p3_r3,p3_r4],dim=1)#256*9=2304       
        
        #256*16=4096 dimension for a picture representation
        predict = torch.cat([p1_t, p2_t, p3_t, p2_at, p3_at], dim=1)

        return (p1_t, p2_t, p3_t, p2_at, p3_at), (p1_g, p2_g, p3_g, p2_fc1, p2_fc2, p2_fc3, p2_fc4, p3_fc1, p3_fc2, p3_fc3, p3_fc4), predict
