import tensorflow

from keras.models import Sequential
from keras.layers import Conv2D
from keras.optimizers import Adam
#from keras.models import to_json


SRCNN = Sequential()

SRCNN.add(Conv2D(filters=128, kernel_size = (9, 9),
				 kernel_initializer='glorot_uniform',
 				activation='relu', padding='valid', use_bias=True, 
 				input_shape=(None, None, 1)))   #None indicates that the dim size is custom


SRCNN.add(Conv2D(filters=64, kernel_size = (3, 3),
				 kernel_initializer='glorot_uniform',
 				activation='relu', padding='same', use_bias=True))

SRCNN.add(Conv2D(filters=1, kernel_size = (5, 5), 
				kernel_initializer='glorot_uniform',
 				activation='linear', padding='valid', use_bias=True))

adam = Adam(lr = 0.0003)

SRCNN.compile(optimizer = adam,
			  loss = 'mean_squared_error',
			  metrics = ['mean_squared_error'])


model_json = SRCNN.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)

