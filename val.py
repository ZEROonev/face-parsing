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

val_dataloader = DataLoader(FaceDataset(path,typel='val'), batch_size=10, num_workers=8)
model=ACSPNet()
#model=FaceParseNet18(num_classes=22,pretrained=True)
model.cuda()
from torchvision.transforms import ToPILImage
import os
import shutil

def makedir(pathd):
  if os.path.exists(pathd):
    shutil.rmtree(pathd)
  os.makedirs(pathd)
makedir('./result/rende/')
makedir('./result/renders/')
color_list = [[0, 0, 0], [204, 0, 0], [76, 153, 0], [204, 204, 0], [51, 51, 255], [204, 0, 204], [0, 255, 255], [255, 204, 204], [102, 51, 0], [255, 0, 0], [102, 204, 0], [255, 255, 0], [0, 0, 153], [0, 0, 204], [255, 51, 153], [0, 204, 204], [0, 51, 0], [255, 153, 51], [0, 204, 0]]
      
def val(name):
    #time_meter = AverageMeter()
    teacher_dict = torch.load(name)
    model.load_state_dict(teacher_dict)
    print('Perform validation...')
    num_samples = len(val_dataloader)
    
    t1 = time.time()
    model.eval()
    metrics = SegMetric(n_classes=22)
    metrics.reset()
    
    for i, data in enumerate(val_dataloader, 0):
        inputs,label,name,h,w= data
        #b,h,w=label.shape
        #h,w=512,512
        inputs=inputs.cuda()
        
        label=label.cuda()
        idx = 0
        interp = torch.nn.Upsample(size=(512,512), mode='bilinear', align_corners=True)
        with torch.no_grad():
            predd = model(inputs)
        
            outputs = predd
            #output = F.interpolate(output, (512, 512), mode='bilinear', align_corners=True)  # [1, 19, 512, 512]
            #parsing = np.squeeze(output.data.max(1)[1].cpu().numpy(), axis=0)
            #img=parsing.astype(np.uint8)
            #out_img = Image.fromarray(img)
            if isinstance(outputs, list):
                for output in outputs:
                    parsing = output
                    
                    parsing = interp(parsing).data.cpu().numpy()
                    
                    pred= np.asarray(np.argmax(parsing, axis=1), dtype=np.uint8)

                    
            else:
                parsing = outputs
                parsing = interp(parsing).data.cpu().numpy()
                #parsing=parsing.data.cpu().numpy()
                #parsing = parsing.transpose(0, 2, 3, 1)  # NCHW NHWC
                pred= np.asarray(np.argmax(parsing, axis=1), dtype=np.uint8)
                
                #
                
                predg=np.squeeze(pred)
                #print(predg.shape)
                '''
                out_img = Image.fromarray(predg)
                im = np.array(out_img)

                im_base = np.zeros((h,w,3))
                
                for idx, color in enumerate(color_list):
                    im_base[im == idx] = color
                result = Image.fromarray((im_base).astype(np.uint8))
                if i%6==0:
                  name=str(eval(str(name).lstrip('(').rstrip(')').rstrip(',')))
                  result.save("./result/renders/%s.png"%(name))
                  out_img.save("./result/rende/%s.png"%(name))
                
                
                '''
                gt=np.squeeze(label.cpu().numpy())
                #print(gt.shape,predg.shape)
                  
                metrics.update(gt, predg)

        
        #time_meter.update(time.perf_counter() - tic)

        # elapsed_time = timeit.default_timer() - start_time
        # print("Inference time: {}fps".format(self.test_size / elapsed_time))
    #print("Inference Time: {:.4f}s".format(time_meter.average() / images.size(0)))

    score = metrics.get_scores()[0]
    class_iou = metrics.get_scores()[1]

    for k, v in score.items():
        print(k, v)

    facial_names = ['background','skin', 'left_eyebrow','right_eyebrow','left_eye', 'right_eye', 
                        
                         'nose', 'upper_lip','inner_mouth', 'lower_lip', 'hair', 
                       'left_eye_shadow','right_eye_shadow','left_ear', 'right_ear','hat','glasses','Else skin']
    for i in range(18):
        print(facial_names[i] + "\t: {}".format(str(class_iou[i])))

      
    
if __name__=='__main__':
  val('./ckpt/face1_31.pth.tea')


    
