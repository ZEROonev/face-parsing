from torch.utils.data import Dataset,DataLoader
import os
import pandas as pd
from skimage import io,transform
import torch
import numpy as np
from PIL import Image,ImageFilter
import csv
from generate_face import crop_to_face
class FaceDataset(Dataset):
  def __init__(self,root_dir,typel,transform=None):
    '''
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
    self.fnames = list(sorted(os.listdir(self.img_path)))
    for item,data in enumerate(self.fnames):
        f=sorted(os.listdir(os.path.join(self.img_path,data)))
        self.total_name+=f
    '''
    self.transform=transform  
    self.typel=typel
    self.total_name=[]
    num=0
    self.test_path='../cvpr/test/image'
    self.test = list(sorted(os.listdir(self.test_path)))
    if self.typel=='test':
      self.num=len(self.test)
    #print(self.test)
    self.names=list(sorted(os.listdir('../cvpr_c/train/image')))
    self.segs=list(sorted(os.listdir('../cvpr_c/train/seg')))
    if self.typel=='train' or self.typel=='val':
      
      self.num=len(self.names)
      #print(self.num)
      #print(len(self.segs))
    '''
    with open('total_name.txt','w') as f:
      f.write(str(self.total_name).lstrip('[').rstrip(']'))
      f.close()
    
    with open('total_name.txt','r') as f:
      self.total_name=list(eval(f.read()))
    '''
    #print(len(self.total_name))
    self.names=list(sorted(os.listdir('../cvpr_c/train/image')))
    self.segs=list(sorted(os.listdir('../cvpr_c/train/seg')))

    # the images in total ,or not total
    
  def __len__(self):
    return self.num

  def __getitem__(self,idx):
    #print(self.total_name)
    if self.typel=='train' or self.typel=='val':

      person_id=self.names[idx]
      person_face=os.path.join('../cvpr_c/train/image',person_id)# return the folder name within the folder
      #img_frame=os.path.join(person_face,person_id)
      #name=str(person_id.split('.')[0])
      image=Image.open(person_face)
      
    
      
      mask_path=os.path.join('../cvpr_c/train/seg',str(person_id.split('.')[0])+'.png')
      mask=Image.open(mask_path).convert("L")
      
      
      
      w,h=mask.size
      #if self.typel=='train':
      #if self.typel=='train' or self.typel=='val':
      image=image.resize((360,360))
      if self.transform:
          image=self.transform(image)
      image=img_transform(image)
      
      if self.typel=='train': 
       
        mask=mask.resize((360,360))
        mask=mask_transform(mask)
        #image=torchvision.transforms.functional.resized_crop(image,0,20,960,960,[512,512])
        #mask=torchvision.transforms.functional.resized_crop(mask,0,10,960,960,[512,512])
        return image,mask
      if self.typel=='val':
        #mask=mask.resize((512,512))
        mask=mask_transform(mask)
        return image,mask,name,h,w

    else:
      im=self.test[idx]
      name=str(im.split('.')[0])
      #print(name)
      image=Image.open(os.path.join(self.test_path,im))
      #image = image.filter(ImageFilter.DETAIL)
      w,h=image.size
      if self.transform:
        image=self.transform(image)
      
      #image=image.resize((512,512))
      image=img_transform(image)
      return image,h,w,name

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