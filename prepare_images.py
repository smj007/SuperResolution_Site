import numpy as np
import os
import math
import cv2
import matplotlib.pyplot as plt
import ntpath
from keras.optimizers import Adam
from keras.layers import Conv2D
from keras.models import Sequential


def prepare_images(path, factor):

	img = cv2.imread(path)

	print(ntpath.basename(path))
	h, w, _ = img.shape
	new_h = h // factor
	new_w = w // factor

	img = cv2.resize(img, (new_w, new_h),
					interpolation = cv2.INTER_LINEAR)

	img = cv2.resize(img, (w, h),
					interpolation = cv2.INTER_LINEAR)

	print('Saving {}'.format(path))
	return img 


def psnr(target, ref):

	target_data = target.astype('float')
	ref_data = ref.astype('float')

	difference = ref_data - target_data
	difference = difference.flatten('C')

	rmse = math.sqrt(np.mean(difference ** 2))

	psnr_value = 20 * math.log10(255. / rmse)

	return psnr_value



def mse(target, ref):

	error = np.sum((target.astype('float') - ref.astype('float')) ** 2)
	error /= float(target.shape[0] * target.shape[1]) #normalization

	return error


def compare_images(target, ref):

	scores = []
	scores.append(psnr(target, ref))
	scores.append(mse(target, ref))

	return scores

