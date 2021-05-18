import torch
#torch.set_printoptions(profile="full")
import torchvision.transforms as transforms
from loader3 import FaceDataset
import torch.optim as optim
import torch.nn as nn
import torch
import time
import torch.nn.functional as F
import tqdm
from torch.autograd import Variable
from model1 import ACSPNet
#from resnet18 import FaceParseNet18
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor,Compose,ColorJitter
#from loss import focal_loss,cross_entropy2d
from aug import *
#define the config
num_epochs=1
resume_epoch=0
path='../cvpr'

teacher=True

from torch.backends import cudnn
cudnn.enabled = True
cudnn.benchmark = True
cudnn.deterministic = False
torch.cuda.manual_seed(2020)

transform = Compose([RandomHorizontallyFlip(p=.5), RandomSized(
            size=512), AdjustBrightness(bf=0.1), AdjustContrast(cf=0.1), AdjustHue(hue=0.1),
                             AdjustSaturation(saturation=0.1)])
trainloader = DataLoader(FaceDataset(path,typel='train'), batch_size=16, shuffle=True, num_workers=8,drop_last=True)

#val_dataloader = DataLoader(FaceDataset(path,typel='val'), batch_size=10, num_workers=2)
model=ACSPNet()

#model=FaceParseNet18(num_classes=22,pretrained=True)
model.load_state_dict(torch.load('./ckpt/face_1.pth.tea'))
criterion=nn.CrossEntropyLoss(ignore_index=255)

optimizer=optim.Adam(model.parameters(),lr=0.0003)
#optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 0.0001, 0.9, 1e-5)

model.cuda()
if teacher:
  teacher_dict = model.state_dict()
criterion.cuda()
#lr_scheduler = WarmupPolyLR(optimizer, max_iters=100*len(trainloader), power=0.9, warmup_factor=1.0 / 3, warmup_iters=1,warmup_method='linear')
def train():
    
    print("Train Start: ")
    for epoch in range(1,100):
   
      print('Epoch %d begin' % (epoch + 1))
      t1 = time.time()

      epoch_loss = 0.0
      model.train()
      #print(len(trainloader))
      for i, data in enumerate(trainloader, 0):

            t2 = time.time()
           
            inputs, labels = data
            _,s,h,w=inputs.shape
            
            inputs = inputs.cuda()
            labels = labels.cuda()
            
            # heat map
            outputs = model(inputs)
            #print(outputs.shape,labels.shape)
            # loss
            #print(labels.shape,labels.long().shape)
            #labels = labels.cuda()
            
            outputs=F.interpolate(input=outputs,size=(h,w))
            #print(outputs.shape,labels.shape)

            #loss = focal_loss(outputs, labels)

            loss = criterion(outputs, labels)
            
            #loss = cross_entropy2d(outputs, labels,reduction='mean')
            # back-prop
            optimizer.zero_grad()
            loss.backward()

            # update param
            optimizer.step()

            #lr_scheduler.step(epoch=None)
            if teacher:
                for k, v in model.state_dict().items():
                    if k.find('num_batches_tracked') == -1:#？？？
                        #print("Use mean teacher")
                        teacher_dict[k] = 0.999 * teacher_dict[k] + (1 - 0.999) * v
                    else:
                        #print("Nullify mean teacher")
                        teacher_dict[k] = 1 * v

            
            batch_loss = loss.item()
          
            
            t3=time.time()
            print('epoch %2d:[%d/%5d]loss:%.3f time:%.3f'%(epoch+1,i+1,len(trainloader),loss.item(),(t3-t2)), end='')
            print()
            epoch_loss+=batch_loss

      
      t4=time.time()
      print("evage_loss %.2f ,time total%.2f"%(epoch_loss/len(trainloader),(t4-t1)))
      filename='./ckpt/%s_%d.pth'%('crop',epoch+1)
      #torch.save(model.state_dict(),filename)
      if teacher:
        torch.save(teacher_dict, filename+'.tea')
'''
def adjust_learning_rate(g_lr, optimizer, i_iter, total_iters):
    """The learning rate decays exponentially"""

    def lr_poly(base_lr, iter, max_iter, power):
        return base_lr * ((1 - float(iter) / max_iter) ** (power))

    lr = lr_poly(g_lr, i_iter, total_iters, .9)
    optimizer.param_groups[0]['lr'] = lr

    return lr 
'''     
if __name__=='__main__':
  train()


    
