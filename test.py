import torch
from loader import FaceDataset
import torch.nn as nn
import torch
import time
from model import ACSPNet
#from resnet18 import FaceParseNet18
from torch.utils.data import DataLoader
import torch.nn.functional as F
import numpy as np
from PIL import Image
from utils import generate_label
from metrics import SegMetric
import cv2 as cv
#define the config
#from f1 import F1Score
#f1_class=F1Score().cuda()
#from loader import FaceDataset
num_epochs=1
resume_epoch=0
path='../cvpr'

val_dataloader = DataLoader(FaceDataset(path,typel='test'), batch_size=1, num_workers=1)

model=ACSPNet()
#model=FaceParseNet18(num_classes=19,pretrained=True)
model.cuda()
from torchvision.transforms import ToPILImage
import os
import shutil
pathd='./seg_test/'

if os.path.exists(pathd):
  shutil.rmtree(pathd)
os.makedirs(pathd)

       
def val(name):
    #time_meter = AverageMeter()
    teacher_dict = torch.load(name)
    model.load_state_dict(teacher_dict)
    print('Perform validation...')
    num_samples = len(val_dataloader)
    print(num_samples)
    
    model.eval()
   
    
    for i, data in enumerate(val_dataloader, 0):
        inputs,h,w,name = data
        
        
        inputs=inputs.cuda()
        
       
       
        interp = torch.nn.Upsample(size=(h,w), mode='bilinear', align_corners=True)
        with torch.no_grad():
            predd = model(inputs)
        
            outputs = predd
            name=str(eval(str(name).lstrip('(').rstrip(')').rstrip(',')))
            
            if isinstance(outputs, list):
                for output in outputs:
                    parsing = output
                    
                    parsing = interp(parsing).data.cpu().numpy()
                    
                    pred= np.asarray(np.argmax(parsing, axis=1), dtype=np.uint8)
                    predg=np.squeeze(pred)
                    
            else:
                parsing = outputs
                parsing = interp(parsing).data.cpu().numpy()
                
                pred= np.asarray(np.argmax(parsing, axis=1), dtype=np.uint8)
                
                predg=np.squeeze(pred)
               
                out_img = Image.fromarray(predg)
                png="./seg_test/%s"%(name.split('_')[0])
                if not os.path.exists(png):
                  os.makedirs(png)
                out_img.save("./seg_test/%s/%s.png"%(name.split('_')[0],name))
                
                #img_path='../cvpr/test/image'
                #img = Image.open(os.path.join(img_path,name+'.jpg')).convert("RGB")
                #fusing = vis_parsing_maps(img, predg)
                #save_path="./result/renders/%s.png"%(name)
                #if not os.path.exists(save_path):
                  #os.makedirs(save_path)
                #fusing.save(save_path="./result/renders/%s.png"%(name))
                
    
if __name__=='__main__':
  val('./ckpt/face1_30.pth.tea')


    
