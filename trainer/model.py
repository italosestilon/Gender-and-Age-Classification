import argparse
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

import cPickle

from tensorflow.python.lib.io import file_io

from keras.applications import vgg16


def get_data(filename):
	train_datagen = ImageDataGenerator(
        rescale=1./255,
        data_format="channels_last",
        horizontal_flip=True)

	train_generator = train_datagen.flow_from_directory(
        filename,
        target_size=(244, 244),
        batch_size=32)

	return train_generator


def define_model(weights_path=None):

	model = Sequential()

	model.add(Flatten())
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(4096, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(57, activation='softmax'))

	if weights_path:
		weights = np.load(weights_path)
		model.set_weights(weights)

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

	model.compile(loss='categorical_crossentropy',
	          optimizer=sgd,
	          metrics=['accuracy'])

	return model

def bottleneck_features(train_dir, batch_size=32):

	model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(244, 244, 3))

	train_datagen = ImageDataGenerator(
        rescale=1./255,
        data_format="channels_last",
        horizontal_flip=True)

	generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(244, 244),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

	bottleneck_features_train = model.predict_generator(generator, 2000)

	with file_io.FileIO('bottleneck_features_train.npy', mode='w+') as output:
		np.save(output, bottleneck_features_train)


def train_model(model, train_generator, steps_per_epoch=3):
	model.fit_generator(train_generator, epochs=1, steps_per_epoch=steps_per_epoch)

	return model

def save_model(model, job_dir):
	model.save('model.h5')
    
	# Save model.h5 on to google storage
	with file_io.FileIO('model.h5', mode='w') as input_f:
		with file_io.FileIO(job_dir + '/model.h5', mode='w+') as output_f:
			output_f.write(input_f.read())

def main():

	parser = argparse.ArgumentParser()
	# Input Arguments
	parser.add_argument(
		'--train-file',
		help='GCS or local paths to training data',
		required=True
	)

	parser.add_argument(
		'--job-dir',
		help='GCS location to write checkpoints and export models',
		required=True
	)
	args = parser.parse_args()
	arguments = args.__dict__
	job_dir = arguments.pop('job_dir')
	train_dir = arguments['train_file']

	train_generator = get_data(train_dir)
	bottleneck_features(train_dir)


	#model = define_model(weights_path='weights/weights.h5py')

	#model = define_model()

	#model = train_model(model, bottleneck_features_generator)
	save_model(model, job_dir)

	#np.save("my_weights", model.get_weights)

if __name__ == "__main__":
	main()
