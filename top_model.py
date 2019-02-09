'''
This file is used to get the Base Model. Models that form the base model are MobileNetV2, InceptionV3, InceptionResNetV2, VGG19, ResNet50 and Xception.
'''

from keras.layers import Dense, Dropout, Lambda, Input
from keras.models import Model
import keras.backend as K


class TensorflowModel():

	def __init__(self, top_model):
	
		self.top_model = top_model
		
		if self.top_model == 'mobilenet':
			from keras.applications.mobilenet_v2 import MobileNetV2 
			self.input_shape = (224, 224)
			self.inp = Input((self.input_shape[0], self.input_shape[1], 3))
			self.initial_model = MobileNetV2(include_top=False, input_shape=(self.input_shape[0], self.input_shape[1], 3), input_tensor=self.inp, pooling='avg')
			
		elif self.top_model == 'inception':
			from keras.applications.inception_v3 import InceptionV3
			self.input_shape = (299, 299)
			self.inp = Input((self.input_shape[0], self.input_shape[1], 3))
			self.initial_model = InceptionV3(include_top=False, input_shape=(self.input_shape[0], self.input_shape[1], 3), input_tensor=self.inp, pooling='avg')
		
		elif self.top_model == 'vgg':
			from keras.applications.vgg19 import VGG19
			self.input_shape = (224, 224)
			self.inp = Input((self.input_shape[0], self.input_shape[1], 3))
			self.initial_model = VGG19(include_top=False, input_shape=(self.input_shape[0], self.input_shape[1], 3), input_tensor=self.inp, pooling='avg')
		
		elif self.top_model == 'xception':
			from keras.applications.xception import Xception
			self.input_shape = (299, 299)
			self.inp = Input((self.input_shape[0], self.input_shape[1], 3))
			self.initial_model = Xception(include_top=False, input_shape=(self.input_shape[0], self.input_shape[1], 3), input_tensor=self.inp, pooling='avg')
		
		elif self.top_model == 'resnet':
			from keras.applications.resnet50 import ResNet50
			self.input_shape = (224, 224)
			self.inp = Input((self.input_shape[0], self.input_shape[1], 3))
			self.initial_model = ResNet50(include_top=False, input_shape=(self.input_shape[0], self.input_shape[1], 3), input_tensor=self.inp, pooling='avg')
		
		elif self.top_model == 'inception-resnet':
			from keras.applications.inception_resnet_v2 import InceptionResNetV2
			self.input_shape = (299, 299)
			self.inp = Input((self.input_shape[0], self.input_shape[1], 3))
			self.initial_model = InceptionResNetV2(include_top=False, input_shape=(self.input_shape[0], self.input_shape[1], 3), input_tensor=self.inp, pooling='avg')
		
		else:
			raise Exception("Values allowed for model parameter are - mobilenet, inception, vgg, xception, resnet and inception-resnet. Value passed was: {}".format(self.top_model))
		
	def _get_model_top(self):
	
		x = Dense(512, activation='relu')(self.initial_model.output)
		x = Dropout(0.3)(x)
		x = Dense(128)(x)
		x = Lambda(lambda x: K.l2_normalize(x, axis=1))(x)

		model_top = Model(inputs=self.inp, outputs=x)
		
		return model_top

	def _get_input_shape(self):
		return self.input_shape