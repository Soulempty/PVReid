import copy
import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50, Bottleneck

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
    
    
    
    
class Fighter(nn.Module):
    def __init__(self,num_classes):
        super(Fighter, self).__init__()

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

        self.maxpool_g = nn.MaxPool2d(kernel_size=(16, 16))
        
        self.maxpool_h = nn.MaxPool2d(kernel_size=(4, 16))
        self.maxpool_v = nn.MaxPool2d(kernel_size=(16, 4))

        #total 3+2+3+4 branch for classificaiton
        self.fc_g_1 = ReductionFc(2048,256,num_classes)
        self.fc_g_2 = ReductionFc(2048,256,num_classes)
        
        self.fc_h_1 = ReductionFc(2048,256,num_classes)
        self.fc_h_2 = ReductionFc(2048,256,num_classes)
        self.fc_h_3 = ReductionFc(2048,256,num_classes)
        self.fc_h_4 = ReductionFc(2048,256,num_classes)

        self.fc_hc_1 = ReductionFc(512,256,num_classes)
        self.fc_hc_2 = ReductionFc(512,256,num_classes)
        self.fc_hc_3 = ReductionFc(512,256,num_classes)
        self.fc_hc_4 = ReductionFc(512,256,num_classes)
        
        self.fc_v_1 = ReductionFc(2048,256,num_classes)
        self.fc_v_2 = ReductionFc(2048,256,num_classes)
        self.fc_v_3 = ReductionFc(2048,256,num_classes)
        self.fc_v_4 = ReductionFc(2048,256,num_classes)
        
        self.fc_vc_1 = ReductionFc(512,256,num_classes)
        self.fc_vc_2 = ReductionFc(512,256,num_classes)
        self.fc_vc_3 = ReductionFc(512,256,num_classes)
        self.fc_vc_4 = ReductionFc(512,256,num_classes)
        #       

    def forward(self, x):
        x = self.backbone(x)

        p1 = self.p1(x)
        p2 = self.p2(x)

        h_g = self.maxpool_g(p1)
        h_g_1 = h_g[:, :512, :, :]
        h_g_2 = h_g[:, 512:1024, :, :]
        h_g_3 = h_g[:, 1024:1536, :, :]
        h_g_4 = h_g[:, 1536:2048, :, :]
        
        v_g = self.maxpool_g(p2)
        v_g_1 = v_g[:, :512, :, :]
        v_g_2 = v_g[:, 512:1024, :, :]
        v_g_3 = v_g[:, 1024:1536, :, :]
        v_g_4 = v_g[:, 1536:2048, :, :]

        h_a = self.maxpool_h(p1)
        h_1 = h_a[:, :, 0:1, :]
        h_2 = h_a[:, :, 1:2, :]
        h_3 = h_a[:, :, 2:3, :]
        h_4 = h_a[:, :, 3:4, :]

        v_a = self.maxpool_v(p2)
        v_1 = v_a[:, :, :, 0:1]
        v_2 = v_a[:, :, :, 1:2]
        v_3 = v_a[:, :, :, 2:3]
        v_4 = v_a[:, :, :, 3:4]
        
        
        #
        trip_g1, h_g_fc = self.fc_g_1(h_g)
        trip_g2, v_g_fc = self.fc_g_2(v_g)
        
        hc_1, hc_1_fc = self.fc_hc_1(h_g_1)
        hc_2, hc_2_fc = self.fc_hc_2(h_g_2)
        hc_3, hc_3_fc = self.fc_hc_3(h_g_3)
        hc_4, hc_4_fc = self.fc_hc_4(h_g_4)
        
        vc_1, vc_1_fc = self.fc_vc_1(v_g_1)
        vc_2, vc_2_fc = self.fc_vc_2(v_g_2)
        vc_3, vc_3_fc = self.fc_vc_3(v_g_3)
        vc_4, vc_4_fc = self.fc_vc_4(v_g_4)
        
        h_1_r, h_1_fc = self.fc_h_1(h_1)
        h_2_r, h_2_fc = self.fc_h_2(h_2)
        h_3_r, h_3_fc = self.fc_h_3(h_3)
        h_4_r, h_4_fc = self.fc_h_4(h_4)
        
        v_1_r, v_1_fc = self.fc_v_1(v_1)
        v_2_r, v_2_fc = self.fc_v_2(v_2)
        v_3_r, v_3_fc = self.fc_v_3(v_3)
        v_4_r, v_4_fc = self.fc_v_4(v_4) 

        predict = torch.cat([h_1_r, h_2_r, h_3_r, h_4_r, v_1_r, v_2_r, v_3_r, v_4_r, hc_1, hc_2, hc_3, hc_4, vc_1, vc_2, vc_3, vc_4, trip_g1, trip_g2], dim=1)

        return (trip_g1, trip_g2), (h_g_fc, v_g_fc, h_1_fc, h_2_fc, h_3_fc, h_4_fc, v_1_fc, v_2_fc, v_3_fc, v_4_fc, hc_1_fc, hc_2_fc, hc_3_fc, hc_4_fc, vc_1_fc, vc_2_fc, vc_3_fc, vc_4_fc), predict
