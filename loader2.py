import cv2
from torch.utils.data import Dataset,DataLoader
import os
import pandas as pd
from skimage import io,transform
import torch
import numpy as np
from PIL import Image,ImageFilter
import csv

class FaceDataset(Dataset):
  def __init__(self,root_dir,typel,transform=None):
    self.root=root_dir
    self.resize_height = 512
    self.resize_width = 512
    self.crop_size = 256
    self.img_typel = os.path.join(root_dir,typel)
    self.img_path = os.path.join(self.img_typel,'image')
    self.seg_path = os.path.join(self.img_typel,'seg')
    self.transform=transform  
    self.typel=typel
    self.total_name=[]
    num=0
    self.knames=[]
    self.fnames = list(sorted(os.listdir(self.img_path)))
    for j in range(len(self.fnames)):
      self.knames+=[self.fnames[j],self.fnames[j],self.fnames[j],self.fnames[j]]

    self.snames = list(sorted(os.listdir(self.seg_path)))
    self.senames=[]
    for j in range(len(self.snames)):
      self.senames+=[self.snames[j],self.snames[j],self.snames[j],self.snames[j]]

    self.test_path='../cvpr/test/image'
    self.test = list(sorted(os.listdir(self.test_path)))
    if self.typel=='test':
      self.num=len(self.test)
   
    if self.typel=='train' or self.typel=='val':
      for item,data in enumerate(self.fnames):
        f=sorted(os.listdir(os.path.join(self.img_path,data)))
        self.total_name+=f
      self.num=len(self.knames)
    
    
  def __len__(self):
    return self.num
  
  def __getitem__(self,idx):
    
    if self.typel=='train' or self.typel=='val':
      
      r=5
      file_dir = os.path.join(self.img_path,self.knames[idx])
      frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
      frame_count = len(frames)
      if idx%4==0:
        frames=frames[:frame_count//4]
      elif idx%4==1:
        frames=frames[frame_count//4:2*frame_count//4]
      elif idx%4==2:
        frames=frames[2*frame_count//4:3*frame_count//4]
      else:
        frames=frames[3*frame_count//4:]
      file_dir1=os.path.join(self.img_path,self.senames[idx])
      frames1 = sorted([os.path.join(file_dir1, img) for img in os.listdir(file_dir1)])
      frame_count1 = len(frames1)
      if idx%4==0:
        frames1=frames1[:frame_count1//4]
      elif idx%4==1:
        frames1=frames1[frame_count1//4:2*frame_count1//4]
      elif idx%4==2:
        frames1=frames1[2*frame_count1//4:3*frame_count1//4]
      else:
        frames1=frames1[3*frame_count1//4:]

      bufferi,w,h,r = self.load_frames(frames,r,idx)
      bufferm,wl,hl = self.load_segs(frames1,r,idx)
      #bufferi,w,h,r = self.load_frames(os.path.join(self.img_path,self.knames[idx]),r,idx)
      #bufferm,wl,hl = self.load_segs(os.path.join(self.seg_path,self.senames[idx]),r,idx)
     
      bufferi = torch.from_numpy(bufferi).permute([0,3,1,2])
      
      bufferm = torch.from_numpy(bufferm).long()
      
      return bufferi,w,h,bufferm 

    else:
      im=self.test[idx]
      name=str(im.split('.')[0])
     
      image=Image.open(os.path.join(self.test_path,im))
      #image = image.filter(ImageFilter.DETAIL)
      w,h=image.size
      if self.transform:
        image=self.transform(image)
      
      #image=image.resize((512,512))
      image=img_transform(image)
      return image,h,w,name
  
  def load_frames(self, frames,r,idx):
      
      #print(file_dir)
      '''
      frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
      frame_count = len(frames)
      if idx%4==0:
        frames=frames[:frame_count//4]
      elif idx%4==1:
        frames=frames[frame_count//4:2*frame_count//4]
      elif idx%4==2:
        frames=frames[2*frame_count//4:3*frame_count//4]
      else:
        frames=frames[3*frame_count//4:]
      '''
      frame_count = len(frames)
      imgq=cv2.imread(frames[0])
      w,h,c=imgq.shape
      #print(imgq.shape)'''
      '''if frame_count<5:
        r=1
      if 5<=frame_count<10:
        r=2
      if 10<=frame_count<12:
        r=2
      if 12<=frame_count<14:
        r=2'''

      
      buffer = np.empty((frame_count,h//r,w//r,3), np.dtype('float32'))
      for i, frame_name in enumerate(frames):
          img=cv2.imread(frame_name)
          img=cv2.resize(img, (w//r, h//r))
          frame = np.array(img).astype(np.float32)
          #print(frame.shape)
          buffer[i] = frame
      #print(buffer.shape)
      return buffer,w//r,h//r,r

  def load_segs(self, frames,r,idx):
      '''
      frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
      frame_count = len(frames)

      if idx%4==0:
        frames=frames[:frame_count//4]
      elif idx%4==1:
        frames=frames[frame_count//4:2*frame_count//4]
      elif idx%4==2:
        frames=frames[2*frame_count//4:3*frame_count//4]
      else:
        frames=frames[3*frame_count//4:]
      '''
      frame_count = len(frames)
      imgq=cv2.imread(frames[0])
      w,h,c=imgq.shape
      '''
      if frame_count<5:
        r=1
      if 5<=frame_count<10:
        r=2
      if 10<=frame_count<12:
        r=2
      if 12<=frame_count<14:
        r=2'''
      buffer = np.empty((frame_count,h//r,w//r), np.dtype('uint8'))
      if frame_count>10:
        framesl=frames[:10]
        framesr=frames[10:]
      for i, frame_name in enumerate(frames):
          img=cv2.imread(frame_name)
          gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
          img=cv2.resize(gray, (w//r, h//r))
          frame = np.array(img).astype(np.uint8)
          buffer[i] = frame
      return buffer,w//r,h//r

  def to_tensor(self, buffer):
      return buffer.transpose(0,3,1,2)


def img_transform(img):
  import torchvision.transforms as transforms
  transformer=transforms.Compose([transforms.ToTensor(),
                  transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
                  ])
  img=transformer(img)
  return img

def mask_transform(segm):
    # to tensor
    segm = torch.from_numpy(np.array(segm)).long()

    return segm
