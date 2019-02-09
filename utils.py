'''
This file has two utility functions, one to augment images and other describes the Learning Rate decay for the model.
'''

import random
import cv2
import math
import numpy as np

def augment_image(img_new):
    
    gamma = random.choice([i for i in np.arange(1.0, 2.5, 0.25)])
    img_new = img_new / 255.
    img_new = img_new ** (1/gamma)
    img_new *= 255
    img_new = img_new.astype('uint8')

    augmentation = np.random.choice(['Flip', 'Rotation', 'None'], p=[0.2, 0.6, 0.2])

    if augmentation == 'Flip':
        img_new = np.fliplr(img_new)

    elif augmentation == 'Rotation':
        angle = random.choice([i for i in range(-5, 5)])

        rows, cols, channels = img_new.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img_new = cv2.warpAffine(img_new, M, (cols, rows))

    else:
        img_new = img_new
    
    return img_new
	

def step_decay(epoch):
   
   initial_lrate = 0.01
   drop = 0.1
   epochs_drop = 1.0
   
   lrate = initial_lrate * math.pow(drop,  
           math.floor((1+epoch)/epochs_drop))
   
   return lrate