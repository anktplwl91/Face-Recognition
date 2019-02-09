'''
This file describes the two loss functions Contrative and Triplet losses used to train the Face recognition models.
'''


import keras.backend as K

margin = 0.4
alpha = 1.0

def contrastive_loss(y_true, y_pred):
	return K.mean((1. - y_true) * K.square(y_pred) + y_true * K.square(K.maximum(margin - y_pred, 0.)))

def triplet_loss(y_true, y_pred):
		
	anchor = y_pred[:, 0:128]
	positive = y_pred[:, 128:256]
	negative = y_pred[:, 256:]

	# distance between the anchor and the positive
	pos_dist = K.sum(K.square(anchor - positive), axis=1)

	# distance between the anchor and the negative
	neg_dist = K.sum(K.square(anchor - negative), axis=1)

	# compute loss
	basic_loss = pos_dist - neg_dist + alpha
	loss = K.maximum(basic_loss, 0.0)
	 
	return loss