import warnings
warnings.filterwarnings("ignore")

import os
import argparse
import random
import glob
import numpy as np
import cv2
from keras.models import Model, load_model
import keras.backend as K
from fr_losses import contrastive_loss, triplet_loss
from face_detection import _get_faces_eyes
import matplotlib.pyplot as plt


def load_contrastive_data(test_dir, model):

	sub_dirs = next(os.walk(test_dir))[1]
	
	choice = random.choice(['same', 'different'])
	
	dir_1 = random.choice(sub_dirs)
	if choice == 'same':
		dir_2 = dir_1
	else:
		while True:
			dir_2 = random.choice(sub_dirs)
			if dir_2 != dir_1: break
			
	img_fi_1 = random.choice(glob.glob(test_dir+"\\"+dir_1+"\\*.jpg"))
	img_fi_2 = random.choice(glob.glob(test_dir+"\\"+dir_2+"\\*.jpg"))
		
	face_1 = _get_faces_eyes(img_fi_1)
	face_2 = _get_faces_eyes(img_fi_2)
			
	TARGET_SIZE = (model.input[0].shape[1], model.input[0].shape[2])
	
	face_1 = cv2.resize(face_1, TARGET_SIZE, cv2.INTER_AREA)
	face_2 = cv2.resize(face_2, TARGET_SIZE, cv2.INTER_AREA)
		
	face_1 = cv2.cvtColor(face_1, cv2.COLOR_BGR2RGB)
	face_2 = cv2.cvtColor(face_2, cv2.COLOR_BGR2RGB)
		
	face_1 = face_1 / 255.
	face_2 = face_2 / 255.
	
	return face_1, face_2


def load_triplet_data(test_dir, model):

	sub_dirs = next(os.walk(test_dir))[1]
	
	dir_1 = random.choice(sub_dirs)
	while True:
		dir_2 = random.choice(sub_dirs)
		if dir_2 != dir_1: break
			
	img_fi_a = random.choice(glob.glob(test_dir+"\\"+dir_1+"\\*.jpg"))
	img_fi_p = random.choice(glob.glob(test_dir+"\\"+dir_1+"\\*.jpg"))
	img_fi_n = random.choice(glob.glob(test_dir+"\\"+dir_2+"\\*.jpg"))
		
	face_a = _get_faces_eyes(img_fi_a)
	face_p = _get_faces_eyes(img_fi_p)
	face_n = _get_faces_eyes(img_fi_n)
			
	TARGET_SIZE = (model.input[0].shape[1], model.input[0].shape[2])
		
	face_a = cv2.resize(face_a, TARGET_SIZE, cv2.INTER_AREA)
	face_p = cv2.resize(face_p, TARGET_SIZE, cv2.INTER_AREA)
	face_n = cv2.resize(face_n, TARGET_SIZE, cv2.INTER_AREA)
		
	face_a = cv2.cvtColor(face_a, cv2.COLOR_BGR2RGB)
	face_p = cv2.cvtColor(face_p, cv2.COLOR_BGR2RGB)
	face_n = cv2.cvtColor(face_n, cv2.COLOR_BGR2RGB)
		
	face_a = face_a / 255.
	face_p = face_p / 255.
	face_n = face_n / 255.

	return face_a, face_p, face_n


def save_image(model_file, test_dir, target_dir, n_images, loss):

	for i in range(n_images):
	
		if loss == "contrastive":
					
			model = load_model(model_file, custom_objects={'contrastive_loss': contrastive_loss})
			
			try:
				face_1, face_2 = load_contrastive_data(test_dir, model)
			except:
				continue
			
			face_tensor_1 = K.expand_dims(face_1, axis=0)
			face_tensor_2 = K.expand_dims(face_2, axis=0)

			output = model.predict_on_batch([face_tensor_1, face_tensor_2])
			
			plt.subplot(121)
			plt.imshow(face_1)
			plt.xticks([])
			plt.yticks([])
			
			plt.text(8, 22, output[0][0], style='italic', fontweight='bold', bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

			plt.subplot(122)
			plt.imshow(face_2)
			plt.xticks([])
			plt.yticks([])
			
			plt.subplots_adjust(wspace=0.01)
			
			plt.savefig(target_dir+"\\result_"+str(i)+".jpg", bbox_inches="tight")

		else:
		
			model = load_model(model_file, custom_objects={'triplet_loss': triplet_loss})

			try:
				face_a, face_p, face_n = load_triplet_data(test_dir, model)
			except:
				continue
			
			face_tensor_a = K.expand_dims(face_a, axis=0)
			face_tensor_p = K.expand_dims(face_p, axis=0)
			face_tensor_n = K.expand_dims(face_n, axis=0)
			
			
			output = model.predict_on_batch([face_tensor_a, face_tensor_p, face_tensor_n])

			plt.subplot(131)
			plt.imshow(face_a)
			plt.xticks([])
			plt.yticks([])
			
			plt.text(8, 22, output[0][0], style='italic', fontweight='bold', bbox={'facecolor':'white', 'alpha':0.5, 'pad':10})

			plt.subplot(132)
			plt.imshow(face_p)
			plt.xticks([])
			plt.yticks([])
			
			plt.subplot(133)
			plt.imshow(face_n)
			plt.xticks([])
			plt.yticks([])

			plt.subplots_adjust(wspace=0.01)
			
			plt.savefig(target_dir+"\\result_"+str(i)+".jpg", bbox_inches="tight")
			

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Making predictions based on saved models')

	parser.add_argument('--test_dir', '-d', help="Directory containing test images for making predictions")
	parser.add_argument('--target_dir', '-t', help="Directory in which predicted results are to be saved")
	parser.add_argument('--model_file', '-m', help="Saved model file which is used to make predictions")
	parser.add_argument('--loss', '-l', help="Loss function used in model")
	parser.add_argument('--num_images', '-n', type=int, help="Number of image pairs to test model accuracy")

	args = parser.parse_args()

	test_dir = args.test_dir
	target_dir = args.target_dir
	model_file = args.model_file
	n_images = args.num_images
	loss = args.loss
	
	if target_dir not in os.listdir("."):
		os.mkdir(target_dir)
	else:
		raise Exception("Directory by name of {} already exists. Give another directory name".format(target_dir))

	save_image(model_file, test_dir, target_dir, n_images, loss)