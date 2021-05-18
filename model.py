import torch
import math
import torch.nn as nn
from l2norm import L2Norm
import torch

import torchvision.models as models

class ACSPNet(nn.Module):
    def __init__(self):
        super(ACSPNet, self).__init__()
        #torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        resnet = models.resnet50(pretrained=True)
        #resnet = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
        #resnet = resnest101(pretrained = True)

        self.conv1 = resnet.conv1
        #self.conv1d = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        self.sn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.p3 = nn.ConvTranspose2d(512, 256, kernel_size=(4,4), stride=(2,2), padding=(1,1))
        self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=(4,4), stride=(4,4), padding=(0,0))
        self.p5 = nn.ConvTranspose2d(2048, 256, (8,8), stride=(8,8), padding=(0,0),dilation=(1,1),output_padding=(0,0))
        
        nn.init.xavier_normal_(self.p3.weight)
        nn.init.xavier_normal_(self.p4.weight)
        nn.init.xavier_normal_(self.p5.weight)
        nn.init.constant_(self.p3.bias, 1e-10)
        nn.init.constant_(self.p4.bias, 1e-10)
        nn.init.constant_(self.p5.bias, 1e-10)

        self.p3_l2 = L2Norm(256, 10)
        self.p4_l2 = L2Norm(256, 10)
        self.p5_l2 = L2Norm(256, 10)

        self.feat = nn.Conv2d(768, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_sn = nn.BatchNorm2d(256)
        self.feat_act = nn.ReLU()

        self.pos_conv = nn.Conv2d(256, 22, kernel_size=1)#输出通道为1
       

        nn.init.xavier_normal_(self.pos_conv.weight)
        
        nn.init.constant_(self.pos_conv.bias, -math.log(0.99/0.01))
        
    def forward(self, x_):
        #print(x_.shape)
       

        x = self.conv1(x_)
        x = self.sn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        
        #print(x.shape)
        x = self.layer2(x)
        p3 = self.p3(x)
        p3 = self.p3_l2(p3)
        #print(x.shape)
        x = self.layer3(x)
        p4 = self.p4(x)
        p4 = self.p4_l2(p4)
        #print(x.shape)
        x = self.layer4(x)
        #print(x.shape)
        p5 = self.p5(x)
        #print(p5.shape)
        p5 = self.p5_l2(p5)
        _,_,h,w=p3.shape
        f=torch.nn.Upsample(size=(h,w), mode='bilinear', align_corners=True)
        #print(p5.shape)
        
        p4=f(p4)
        p5=f(p5)
        #print(p3.shape,p4.shape,p5.shape)
        
        cat = torch.cat([p3, p4, p5], dim=1)

        feat = self.feat(cat)
        feat = self.feat_sn(feat)
        feat = self.feat_act(feat)

        x_cls = self.pos_conv(feat)
        #print(x_cls.shape)
        
        #x_cls = torch.sigmoid(x_cls)
        
        return x_cls
