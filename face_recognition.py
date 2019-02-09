import os
import logging
import random
import glob
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate, Lambda
from keras.applications import MobileNetV2
import keras.backend as K




BATCH_SIZE = 10
TARGET_SIZE = (224, 224)


root_dir = 'C:\\Users\\apaliwal\\AppData\\Local\\Continuum\\Anaconda3\\Lib\\site-packages\\cv2\\data\\'
face_cascade = cv2.CascadeClassifier(root_dir + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(root_dir + 'haarcascade_eye.xml')

sub_dirs = ['faces94', 'faces95', 'faces96', 'grimace']

all_dirs = []
for d in sub_dirs:
    subs = next(os.walk(d))[1]
    all_dirs.extend([d+'\\'+s for s in subs])

all_dirs = list(set(all_dirs))


train_dirs = random.sample(all_dirs, int(0.85 * len(all_dirs)))
valid_dirs = list(set(all_dirs) - set(train_dirs))


def _get_augmented_images(img):
    
    gamma = random.choice([i for i in np.arange(1.0, 2.5, 0.25)])
    
    img = img / 255.
    img = img ** (1/gamma)
    img *= 255
    img = img.astype('uint8')

    augmentation = np.random.choice(['Flip', 'Rotation', 'None'], p=[0.2, 0.6, 0.2])
    
    if augmentation == 'Flip':
        img = np.fliplr(img)

    elif augmentation == 'Rotation':

        angle = random.choice([i for i in range(-10, 10, 2)])

        rows, cols, channels = img.shape
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        img = cv2.warpAffine(img, M, (cols, rows))

    else:
        img = img

    return img


def _get_faces(img_fi):
    
    img = cv2.imread(img_fi)        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    for (x,y,w,h) in faces:

        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi = img[y:y+h, x:x+w]
    
    roi = cv2.resize(roi, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    roi = _get_augmented_images(roi)
    
    return roi


def _create_image_pairs(batch_size=BATCH_SIZE, mode_list=train_dirs):
    
    while True:
        
        X1 = []
        X2 = []
        y = []
        
        for idx in range(batch_size // 2):
            
            d = random.choice(mode_list)
            d_imgs = glob.glob(d+'\\*')
            
            
            try:
                img_fi_1, img_fi_2 = random.sample(d_imgs, 2)
                face_1 = _get_faces(img_fi_1)
                face_2 = _get_faces(img_fi_2)
            
            except:
                idx = idx - 1
            
            face_1 = face_1 / 255.
            face_2 = face_2 / 255.
            
            X1.append(face_1)
            X2.append(face_2)
            y.append(0.0)

        for idx in range(batch_size // 2):
            
            d1, d2 = random.sample(mode_list, 2)
            d1_imgs = glob.glob(d1+'\\*')
            d2_imgs = glob.glob(d2+'\\*')
            
            try:
                img_fi_1 = random.choice(d1_imgs)
                img_fi_2 = random.choice(d2_imgs)
                
                face_1 = _get_faces(img_fi_1)
                face_2 = _get_faces(img_fi_2)
                
            except:
                idx = idx - 1
            
            face_1 = face_1 / 255.
            face_2 = face_2 / 255.
            
            X1.append(face_1)
            X2.append(face_2)
            y.append(1.0)


        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        y = np.asarray(y)
                
        yield [X1, X2], y


train_gen = _create_image_pairs()
valid_gen = _create_image_pairs(mode_list=valid_dirs)


def euclidean_dist(inputs):
    
    assert len(inputs)==2, 'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    
    u, v = inputs
    return K.sqrt(K.sum(K.square(u - v), axis=1, keepdims=True))


def contrastive_loss(y_true, y_pred):
    
    margin=1.
    return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))


inp = Input((224, 224, 3))

mobile_model = MobileNetV2(include_top=False, input_shape=(224, 224, 3), input_tensor=inp, pooling='avg')
x = Dense(512, activation='relu')(mobile_model.output)
x = Dropout(0.3)(x)
x = Dense(128)(x)
x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)

model_top = Model(inputs=inp, outputs=x)
#model_top.summary()


inp_1 = Input((224, 224, 3))
inp_2 = Input((224, 224, 3))

out_1 = model_top(inp_1)
out_2 = model_top(inp_2)

merge_layer = Lambda(euclidean_dist)([out_1, out_2])

model = Model(inputs=[inp_1, inp_2], outputs=merge_layer)
#model.summary()

model.compile(loss=contrastive_loss, optimizer='rmsprop')

history = model.fit_generator(train_gen, steps_per_epoch=(len(train_dirs)*20)//BATCH_SIZE, epochs=1, validation_data=valid_gen, validation_steps=(len(valid_dirs)*20)//BATCH_SIZE)