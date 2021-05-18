import face_recognition
import cv2
import numpy as np


def crop_to_face(im,mask):
  im_array = np.asarray(im)
  
  try:
    face_location = face_recognition.face_locations(im_array)
  except:
    print("!!! ERROR: Might be an unsuported image !!! ")
    return None


  #If no face was detected
  if not face_location:
    #print("Didn't find a face")
    #display(im)
    return None,None

  top, right, bottom, left = face_location[0]
  box_size = abs(bottom-top)
  #print(box_size)
  padding = box_size*0.6
  im_cropped = im.crop((left-padding,top-padding, right+padding, bottom+padding))
  se_cropped = mask.crop((left-padding,top-padding, right+padding, bottom+padding))
  imf=im_cropped.resize((420,420))
  mas=se_cropped.resize((420,420))




  return imf,mas





