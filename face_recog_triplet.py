'''
This python file is used for Face Recognition using Triplet Loss. The idea is to generate batches of triplet of images, one Anchor image, other Positive image i.e. same to Anchor, and  
last Negative image i.e. different to Anchor. The Triplet Loss, described in https://arxiv.org/pdf/1503.03832.pdf , is a distance-based loss function. It works by training the model by
simultaneously decreasing euclidean distance between the Anchor and Positive images and increasing the euclidean distance between the Anchor and Negative images.
'''

import warnings
warnings.filterwarnings("ignore")


import os
import random
import argparse
import glob
import numpy as np
import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate, Lambda
from keras import optimizers
from keras.callbacks import LearningRateScheduler, EarlyStopping
import keras.backend as K
from top_model import TensorflowModel
from fr_losses import triplet_loss
from utils import augment_image, step_decay


def _get_image_batches(batch_size, list_name, img_size):

    while True:
        
        img_a = []
        img_p = []
        img_n = []
        y_dummy = []
        
        for _ in range(batch_size):
            
            person_1, person_2 = random.sample(list_name, 2)

            image_files_1 = glob.glob(person_1+"/*.jpg")
            image_files_2 = glob.glob(person_2+"/*.jpg")

            img_fi_a = random.choice(image_files_1)
            img_fi_p = random.choice(image_files_1)
            img_fi_n = random.choice(image_files_2)
            
            face_a = cv2.imread(img_fi_a)
            face_p = cv2.imread(img_fi_p)
            face_n = cv2.imread(img_fi_n)
            
            face_a = cv2.cvtColor(face_a, cv2.COLOR_BGR2RGB)
            face_p = cv2.cvtColor(face_p, cv2.COLOR_BGR2RGB)
            face_n = cv2.cvtColor(face_n, cv2.COLOR_BGR2RGB)
			
            face_a = cv2.resize(face_a, img_size, cv2.INTER_AREA)
            face_p = cv2.resize(face_p, img_size, cv2.INTER_AREA)
            face_n = cv2.resize(face_n, img_size, cv2.INTER_AREA)
			
			
            if list_name == train_people:
                face_a = augment_image(face_a)
                face_p = augment_image(face_p)
                face_n = augment_image(face_n)
            
            face_a = face_a / 255.
            face_p = face_p / 255.
            face_n = face_n / 255.
            
            img_a.append(face_a)
            img_p.append(face_p)
            img_n.append(face_n)
            y_dummy.append(0.)
            
        img_a = np.array(img_a)
        img_p = np.array(img_p)
        img_n = np.array(img_n)
        y_dummmy = np.array(y_dummy)
        
        yield [img_a, img_p, img_n], y_dummy



def _get_complete_model(model_top, input_shape):

	inp_a = Input((input_shape[0], input_shape[1], 3))
	inp_p = Input((input_shape[0], input_shape[1], 3))
	inp_n = Input((input_shape[0], input_shape[1], 3))

	out_a = model_top(inp_a)
	out_p = model_top(inp_p)
	out_n = model_top(inp_n)

	concat_layer = Concatenate()([out_a, out_p, out_n])
	
	model = Model(inputs=[inp_a, inp_p, inp_n], outputs=concat_layer)
	
	return model


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Face Recognition using Triplet Loss')
	
	parser.add_argument('--batch_size', '-b', type=int, default=16, help="Batch Size taken by model, defaults to 16")
	parser.add_argument('--img_dir', '-d', help="Root directory path containing faces of all persons")
	parser.add_argument('--model', '-m', default='mobilenet', help="Pre-trained tensorflow model to be used")
	parser.add_argument('--optim', '-o', default='sgd', help="Optimizer needed to train the model, defaults to SGD")
	parser.add_argument('--epochs', '-e', type=int, default=5, help="Number of epochs model should run for")
	
	
	args = parser.parse_args()
	
	
	BATCH_SIZE = args.batch_size
	img_dir = args.img_dir
	n_epochs = args.epochs
	
	
	if args.optim == "sgd":
		optimizer = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
		
	elif args.optim == "rmsprop":
		optimizer = optimizers.RMSprop(lr=0.01)
		
	elif args.optim == "adam":
		optimizer = optimizers.Adam(lr=0.01)
	
	else:
		raise Exception("Optimizers allowed are - sgd, rmsprop and adam. Optimizer provided was {}". format(args.optim))
	
	
	lrate = LearningRateScheduler(step_decay)
	early_stop = EarlyStopping(monitor="val_loss", patience=3, verbose=1)
	
	t_model = TensorflowModel(args.model)
	model_top = t_model._get_model_top()
	input_shape = t_model._get_input_shape()
	
	person_dirs = []
	
	sub_d = next(os.walk(img_dir))[1]
	person_dirs.extend([img_dir+"\\"+sd for sd in sub_d])

	train_people = random.sample(person_dirs, int(0.85 * len(person_dirs)))
	valid_people = list(set(person_dirs) - set(train_people))
	
	train_gen = _get_image_batches(BATCH_SIZE, list_name=train_people, img_size=input_shape)
	valid_gen = _get_image_batches(BATCH_SIZE, list_name=valid_people, img_size=input_shape)


	model = _get_complete_model(model_top, input_shape)
	model.compile(loss=triplet_loss, optimizer=optimizer)


	history = model.fit_generator(train_gen, steps_per_epoch=(len(train_people)*10)//BATCH_SIZE, epochs=n_epochs, validation_data=valid_gen, validation_steps=(len(valid_people)*10)//BATCH_SIZE, callbacks=[lrate, early_stop])
	model.save("triplet_"+args.model+"_model.h5")