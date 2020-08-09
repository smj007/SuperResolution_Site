import numpy as np
import math
import os

from tensorflow.keras.models import model_from_json
import cv2
import matplotlib.pyplot as plt 
from prepare_images import psnr,mse,compare_images,prepare_images


def modcrop(img, scale):  #to crop and remove the borders from the image

	temp_size = img.shape
	size = temp_size[0:2]
	size = size - np.mod(size, scale)
	img = img[0:size[0], 1:size[1]]
	return img   


def shave(image, border):

	img = image[border : -border, border : -border]
	return img 


def predict(image, img_name):

	with open('model.json', "r") as json_file:
		loaded_model_json = json_file.read()

	SRCNN = model_from_json(loaded_model_json)
	SRCNN.load_weights('3051crop_weight_200.h5')	

	degraded = image
	file = img_name
	ref = cv2.imread('static/input/{}'.format(file))

	ref = modcrop(ref, 3)
	degraded = modcrop(degraded, 3)
	#imgs are processed with modcrop

	# convert the image to YCrCb - (SRCNN trained on Y channel)

	temp = cv2.cvtColor(ref, cv2.COLOR_BGR2YCrCb)

	#create img slice and normalize
	Y = np.zeros((1, temp.shape[0], temp.shape[1], 1), dtype = float)
	Y[0, :, :, 0] = temp[:, :, 0].astype(float) / 255


	#super-resolution
	prediction = SRCNN.predict(Y, batch_size = 1)

	#Post-processing
	prediction *= 255
	prediction[prediction[:] > 255] = 255
	prediction[prediction[:] < 0] = 0
	prediction = prediction.astype(np.uint8)

	#copy Y channel back to image and convert to BGR
	temp = shave(temp, 6)
	temp[:, :, 0] = prediction[0, :, :, 0]
	output = cv2.cvtColor(temp, cv2.COLOR_YCrCb2BGR)

	#remove the border from the reference and degraded img
	ref = shave(ref.astype(np.uint8), 6)
	degraded = shave(degraded.astype(np.uint8), 6)

	#img quality calculation
	scores = []
	scores.append(compare_images(degraded, ref))
	scores.append(compare_images(output, ref))


	return ref, degraded, output, scores

