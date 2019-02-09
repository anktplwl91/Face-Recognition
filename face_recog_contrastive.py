'''
This python file is used for Face Recognition using Contrastive Loss. The idea is to generate batches of pair of images, both same and different, along with appropriate label, a pair of face images 
of same person has label 0 and that of different people has label 1. The Contrastive Loss, described in http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf , is a distance-based loss
function. It works by training the model by decreasing the similar instances and increasing the distance between dis-similar instances.
'''
import warnings
warnings.filterwarnings("ignore")


import os
import argparse
import random
import glob
import numpy as np
import cv2
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Concatenate, Lambda
from keras import optimizers
from keras.callbacks import LearningRateScheduler, EarlyStopping
import keras.backend as K
from top_model import TensorflowModel
from fr_losses import contrastive_loss
from utils import augment_image, step_decay


def _get_image_batches(batch_size, list_name, img_size):
    
    while True:
        
        X1 = []
        X2 = []
        y = []
        
        for _ in range(batch_size):
            
            choice = random.choice(['same', 'different'])
            
            if choice == 'same':
                person_1 = random.choice(list_name)
                person_2 = person_1
                y.append(0.)
            
            else:
                person_1, person_2 = random.sample(list_name, 2)
                y.append(1.)
                        
            image_files_1 = glob.glob(person_1+"\\*.jpg")
            image_files_2 = glob.glob(person_2+"\\*.jpg")
            
            img_fi_1 = random.choice(image_files_1)
            img_fi_2 = random.choice(image_files_2)
            
            face_1 = cv2.imread(img_fi_1)
            face_2 = cv2.imread(img_fi_2)
			
            face_1 = cv2.cvtColor(face_1, cv2.COLOR_BGR2RGB)
            face_2 = cv2.cvtColor(face_2, cv2.COLOR_BGR2RGB)
		
            face_1 = cv2.resize(face_1, img_size, cv2.INTER_AREA)
            face_2 = cv2.resize(face_2, img_size, cv2.INTER_AREA)
            
            if list_name == train_people:
                face_1 = augment_image(face_1)
                face_2 = augment_image(face_2)
            
            face_1 = face_1 / 255.
            face_2 = face_2 / 255.
            
            X1.append(face_1)
            X2.append(face_2)
            
        X1 = np.asarray(X1)
        X2 = np.asarray(X2)
        y = np.asarray(y)
        
        yield [X1, X2], y


def euclidean_dist(inputs):
    
    assert len(inputs) == 2, 'Euclidean distance needs 2 inputs, %d given' % len(inputs)
    
    u, v = inputs
    return K.sqrt(K.sum(K.square(u - v), axis=1, keepdims=True))


def _get_complete_model(model_top, input_shape):

	inp_1 = Input((input_shape[0], input_shape[1], 3))
	inp_2 = Input((input_shape[0], input_shape[1], 3))

	out_1 = model_top(inp_1)
	out_2 = model_top(inp_2)

	merge_layer = Lambda(euclidean_dist)([out_1, out_2])

	model = Model(inputs=[inp_1, inp_2], outputs=merge_layer)
	
	return model


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Face Recognition using Contrastive Loss')
	
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
	model.compile(loss=contrastive_loss, optimizer=optimizer)


	history = model.fit_generator(train_gen, steps_per_epoch=(len(train_people)*10)//BATCH_SIZE, epochs=n_epochs, validation_data=valid_gen, validation_steps=(len(valid_people)*10)//BATCH_SIZE, callbacks=[lrate, early_stop])
	model.save("contrastive_"+args.model+"_model.h5")