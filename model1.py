import torch
import math
import torch.nn as nn
from l2norm import L2Norm
import torch
from resnest.resnest import resnest50
from torch.nn import functional as F
import torchvision.models as models

class ACSPNet(nn.Module):
    def __init__(self):
        super(ACSPNet, self).__init__()
        #torch.hub.list('zhanghang1989/ResNeSt', force_reload=True)
        resnet = models.resnet50(pretrained=True)
        #resnet = models.resnet50_vd_ssld(pretrained=True)
        #resnet = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
        #resnet = resnest50(pretrained = True)

        self.conv1 = resnet.conv1
        self.conv1d = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
        self.sn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        

        self.p3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=2)
        self.p4 = nn.ConvTranspose2d(1024, 256, kernel_size=4, stride=4, padding=2)
        self.p5 = nn.ConvTranspose2d(2048, 256, kernel_size=8, stride=8, padding=4)
        
        nn.init.xavier_normal_(self.p3.weight)
        nn.init.xavier_normal_(self.p4.weight)
        nn.init.xavier_normal_(self.p5.weight)
        nn.init.constant_(self.p3.bias, 1e-10)
        nn.init.constant_(self.p4.bias, 1e-10)
        nn.init.constant_(self.p5.bias, 1e-10)

        self.p3_l2 = L2Norm(256, 10)
        self.p4_l2 = L2Norm(256, 10)
        self.p5_l2 = L2Norm(256, 10)

        self.feat = nn.Conv2d(768+32, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.feat_sn = nn.BatchNorm2d(256)
        self.feat_act = nn.ReLU()

        self.pos_conv = nn.Conv2d(256, 30, kernel_size=1)#输出通道为1
       

        nn.init.xavier_normal_(self.pos_conv.weight)
        
        nn.init.constant_(self.pos_conv.bias, -math.log(0.99/0.01))
        
        self.brancn_sn3 = nn.BatchNorm2d(3)
        self.branch_relu3 = nn.ReLU()

        self.branch_sn = nn.BatchNorm2d(3)
        self.branch_relu = nn.ReLU()   
        self.brancn_sn1 = nn.BatchNorm2d(3)
        self.branch_relu1 = nn.ReLU()
        self.can=nn.Conv2d(6,32,3,stride=2,padding=1)
        self.brancn_sn2 = nn.BatchNorm2d(32)
        self.branch_relu2 = nn.ReLU()

        
    def forward(self, x_):
        #print(x_.shape)
        sobel = [[0, 1, 0], [1,-4,1], [0, 1, 0]]
        sobel_kernel = torch.cuda.FloatTensor(sobel).expand(3,3,3,3)
        xi=F.conv2d(x_, sobel_kernel,stride=1,padding=1)
        xi.bias=None
        xi=self.brancn_sn3(xi)
        xi=self.branch_relu3(xi)
        #print(xi.shape)
        x_image=torch.cat([xi,x_],dim=1)

        #x = self.conv1(x_image)
        x = self.conv1(x_)
        x = self.sn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        

        x = self.layer2(x)
        p3 = self.p3(x)
        p3 = self.p3_l2(p3)

        x = self.layer3(x)
        p4 = self.p4(x)
        p4 = self.p4_l2(p4)

        x = self.layer4(x)
        p5 = self.p5(x)
        p5 = self.p5_l2(p5)

        sobel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        sobel_kernel = torch.cuda.FloatTensor(sobel).expand(3,3,3,3)
        x2=F.conv2d(x_, sobel_kernel,stride=2, padding=1)
        x2.bias=None
        x2=self.branch_sn(x2)
        x2=self.branch_relu(x2)
       
        sobel_1 = [[-1,0,1],[-2,0,2],[-1,0,1]]
        sobel_kernel_1 = torch.cuda.FloatTensor(sobel_1).expand(3,3,3,3)
        x3=F.conv2d(x_,sobel_kernel_1,stride=2,padding=1)
        #torch.nn.functional.conv2d()
        #conv3.weight.data=sobel_kernel_1
        x3.bias=None
        
        x3=self.brancn_sn1(x3)
        x3=self.branch_relu1(x3)
        cat_s=torch.cat([x2,x3],dim=1)
        cat_s=self.can(cat_s)
        cat_s=self.brancn_sn2(cat_s)
        cat_s=self.branch_relu2(cat_s)

       
        _,_,h,w=p3.shape
        f=torch.nn.Upsample(size=(h,w), mode='bilinear', align_corners=True)
        #print(p5.shape)
        if p4.shape!=p3.shape:
          p4=f(p4)
        if p5.shape!=p3.shape:
          p5=f(p5)
        if cat_s.shape!=p3.shape:
          cat_s=f(cat_s)

        cat = torch.cat([p3, p4, p5,cat_s], dim=1)

        feat = self.feat(cat)
        feat = self.feat_sn(feat)
        feat = self.feat_act(feat)

        x_cls = self.pos_conv(feat)
        #print(x_cls.shape)
        
        #x_cls = torch.sigmoid(x_cls)
        
        return x_cls
