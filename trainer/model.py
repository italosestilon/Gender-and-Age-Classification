################################################################
# Project: Neural Network for age and gender estimation 
# Authors Italos __complete me__ Estilon
#         João Phillipe Cardenuto 
# Universidade Estadual de Campinas
#
#'''
#2017 (c) MIT License
#
#TODO:
#
#
################################################################
import argparse
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from io import BytesIO

import cPickle
import pickle
import math

from tensorflow.python.lib.io import file_io

from keras.applications import vgg16

def get_data(train_dir, batch_size=32, input_shape=(32, 32, 3), shuffle=False, job_type=1):

	with file_io.FileIO(train_dir +"/id.txt", mode='r') as input_fn:
		ids = pickle.load(input_fn)
	
	generator = DataGenerator(dim_x = input_shape[0], dim_y = input_shape[1], dim_z = input_shape[2], batch_size = batch_size, shuffle=shuffle, train_dir=train_dir, data_type=job_type)

	train_generator = generator.generate(ids)

	return train_generator, ids


def define_model(weights_path=None, input_shape=(32,32,3)):

	model = Sequential()
	model.add(Flatten(input_shape=input_shape))
	model.add(Dense(input_shape[1]*input_shape[2], activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(input_shape[1]*input_shape[2], activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))

	if weights_path:
		weights = np.load(weights_path)
		model.set_weights(weights)

	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

	model.compile(loss='categorical_crossentropy',
	          optimizer=sgd,
	          metrics=['accuracy'])

	return model

def bottleneck_features(train_dir, batch_size=32, number_of_samples=20000, input_shape=(1,32,32), output_dir="vgg_preditc", job_type=1):

	model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)

	#generator = DataGenerator(dim_x = input_shape[0], dim_y = input_shape[1], dim_z = input_shape[2], batch_size = batch_size, shuffle = False, train_dir=train_dir)

	predict_generator, ids = get_data(train_dir, batch_size=batch_size, input_shape=input_shape, job_type=job_type)

	j = 0
	n = int(math.ceil(number_of_samples/float(batch_size)))
	for i in range(n):

		print("Predicting batch {}/{}".format(i, n))

		X_batch, y_batch = predict_generator.next()

		y = model.predict(X_batch)

		for sample in range(y.shape[0]):
                    try:
                        with file_io.FileIO(output_dir+'/'+str(np.argmax(y_batch[sample])).zfill(2)+'/' + ids[j] , mode='w+') as output:
                            np.save(output, y[sample])
                            j = j + 1
                    except: 
                        print "Warning one batch is not FULL"
                        break


def train_model(model, train_generator, epochs=20, steps_per_epoch=100):
	model.fit_generator(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)

	return model

def save_model(model, job_dir):
	model.save('model.h5')
    
	# Save model.h5 on to google storage
	with file_io.FileIO('model.h5', mode='r') as input_f:
		with file_io.FileIO(job_dir + '/model.h5', mode='w') as output_f:
			output_f.write(input_f.read())


class DataGenerator(object):
	'Generates data for Keras'
	def __init__(self, dim_x = 32, dim_y = 32, dim_z = 32, batch_size = 32, shuffle = False, train_dir=None, data_type=1):
		'Initialization'
		self.dim_x = dim_x
		self.dim_y = dim_y
		self.dim_z = dim_z
		self.batch_size = batch_size
		self.shuffle = shuffle
		self.data_type = data_type

		if train_dir == None:
			print "Thunderfuck passou aqui! Deu erro abestado. Ai dento!! Iiiihhii!"
			raise ValueError

		self.train_dir = train_dir

	def generate(self, list_IDs):
		self.list_IDs = list_IDs
		'Generates batches of samples'
		# Infinite loop
		while 1:
			# Generate order of exploration of dataset
			indexes = self.__get_exploration_order(list_IDs)
		
			# Generate batches
			imax = int(math.ceil(len(indexes)/float(self.batch_size)))

			for i in range(imax):
				#print("i generate", i)
				# Find list of IDs
				list_IDs_temp = [list_IDs[k] for k in indexes[i*self.batch_size:(i+1)*self.batch_size]]

				# Generate data
				X, y = self.__data_generation(list_IDs_temp)

				yield X, y

	def __get_exploration_order(self, list_IDs):
		'Generates order of exploration'
		# Find exploration order
		indexes = np.arange(len(list_IDs))
		if self.shuffle == True:
			np.random.shuffle(indexes)

		return indexes

	def __data_generation(self, list_IDs_temp):
		'Generates data of batch_size samples' # X : (n_samples, v_size, v_size, v_size, n_channels)
		# Initialization
		X = np.empty((self.batch_size, self.dim_x, self.dim_y, self.dim_z))
		y = np.empty((self.batch_size), dtype = int)

		if(self.data_type == 0):
			from skimage import io
		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			#print(ID)
			#print("Entrou para pegar dados do bucket")
			#print(ID.split('.')[0])
			#f = BytesIO(file_io.read_file_to_string(self.train_dir +"/" + str(ID[0]) + "/" + ID))
			# Store volume
			if(self.data_type == 1):
				f = BytesIO(file_io.read_file_to_string(self.train_dir +"/" + str(ID[0:2]) + "/" + ID))
				X[i, :, :, :] = np.load(f)
			elif(self.data_type == 0):
				X[i, :, :, :] = io.imread(self.train_dir +"/" + str(ID[0:2]) + "/" + ID)

			# Store class
			y[i] = int(ID[0:2])

			#print("joao sucks",y[i])
		#print("Retornando um batch")
		return X, self.sparsify(y)

	def sparsify(self, y):
		'Returns labels in binary NumPy array'
		n_classes = 2 # Enter number of classes
		return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
			for i in range(y.shape[0])])
		#print("terminou de sparsify", y)

def discover_num_samples(train_dir = None):
    if train_dir == None:
        raise IOError("File id.txt Variable train_dir not initialized")
    try:
        num_samples = np.load(train_dir+"/id.txt")
    except IOError:
        print "File id.txt not Found, or not in pickle.dump type"
    else:
        return len(num_samples)

def discover_input_shape(train_dir = None):
    if train_dir == None:
        raise IOError("File id.txt Variable train_dir not initialized")
    try:
        num_samples = np.load(train_dir+"/id.txt")
    except IOError:
        print "File id.txt not Found, or not in pickle.dump type"
    else:

        sample_path = num_samples[0]
        sample = np.load(train_dir+"/"+sample_path[0:2]+"/"+sample_path)
        return sample.shape


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

	parser.add_argument(
		'--job-type',
		help='The type of job. It is 1 if it a normal job. It is 0 if the job is for getting the predict from vgg16.',
		required=True
	)

	parser.add_argument(
		'--predict-dir',
		help='Dir to save the predict form vgg',
		required=False
	)

	parser.add_argument(
		'--batch-size',
		help='Numbers of batch for NN train',
		required=False
	)

	parser.add_argument(
		'--epochs',
		help='Number of epochs in training',
		required=False
	)
	parser.add_argument(
		'--epochs-steps',
		help='Number of steps in each epoch',
		required=False
	)

	args = parser.parse_args()
	arguments = args.__dict__
	job_dir = arguments.pop('job_dir')
	train_dir = arguments['train_file']
	job_type = arguments['job_type']

        if(arguments['batch_size']):
            batch_size = int(arguments['batch_size'])
        else:
            print "(W) Batch_size not defined and will be set equal 32"
            batch_size=32
        print "Batch Size:",batch_size
        #train_generator = get_data(train_dir)
	

	if(job_type == "1"):
                
                if(arguments['epochs']):
                    epochs = int(arguments['epochs'])
                else:
                    epochs = 10
                if(arguments['epochs_steps']):
                    steps_per_epoch = int(arguments['epochs_steps'])
                else:
                    steps_per_epoch= 100
                input_shape = discover_input_shape(train_dir)
                print "Input Shape:",input_shape
		model = define_model(input_shape=input_shape)
		train_generator, _ = get_data(train_dir, batch_size=batch_size, input_shape=input_shape, shuffle=True)
		model = train_model(model, train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch)
		save_model(model, job_dir)

	elif(job_type == "0"):
		if(arguments['predict_dir']):
                    number_of_samples = discover_num_samples(train_dir)
                    output_predict = arguments['predict_dir']
                    bottleneck_features(train_dir, batch_size=batch_size, number_of_samples=number_of_samples, input_shape=(218, 178, 3), output_dir=output_predict, job_type=int(job_type))
		else:
			print("The predict output dir has not been provided.")

			raise ValueError


	else:
		print("Invalid job type.")
		raise ValueError
	

	#np.save("my_weights", model.get_weights)[]


if __name__ == "__main__":
	main()
