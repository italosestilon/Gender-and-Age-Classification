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
################################################################
import argparse
from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.models import model_from_json
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

from keras import callbacks
from io import BytesIO

import pickle
import math

from tensorflow.python.lib.io import file_io

from keras.applications import vgg16

def get_data(train_dir, batch_size=32, input_shape=(32, 32, 3), shuffle=False, data_type=1):

    with file_io.FileIO(train_dir +"/id.txt", mode='r') as input_fn:
        ids = pickle.load(input_fn)
    
    generator = DataGenerator(dim_x = input_shape[0], dim_y = input_shape[1], dim_z = input_shape[2], batch_size = batch_size, shuffle=shuffle, train_dir=train_dir, data_type=data_type)

    train_generator = generator.generate(ids)

    return train_generator, ids


def define_model(weights_path=None, input_shape=(32,32,3)):

    model = Sequential()
    model.add(Flatten(input_shape=input_shape))

    model.add(Dense(4096, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))

    if weights_path:
        weights = np.load(weights_path)
        model.set_weights(weights)

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

    return model

# Forward on the bottleneck NN
def bottleneck_features(train_dir, batch_size=32, number_of_samples=20000, input_shape=(1,32,32), \
        output_dir="vgg_preditc", job_type=1 ,passing_model = None):

    if(job_type == 1 or job_type == 0):
        model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
    elif (job_type == 2 and passing_model ):
        model = passing_model
    else:
        print (" Error type_job in bottleneck_features function or model not defined")
        raise ValueError

    predict_generator, ids = get_data(train_dir, batch_size=batch_size, input_shape=input_shape, data_type=job_type)
    j = 0
    error = 0
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
                if(error):
                    print("Dir not Found")
                    raise
                print("Warning one batch is not FULL, or dir not found")
                error+=1

                break


def train_model(model, train_generator, epochs=20, steps_per_epoch=100,validation_data=None,validation_steps=None, output_dir=None):
    callbacks_ = None
    if output_dir is not None:
        callbacks_ = [callbacks.ModelCheckpoint(output_dir+"/weights.{epoch:02d}-{val_loss:.2f}.hdf5", monitor='val_loss', 
            verbose=1, save_best_only=False,
             save_weights_only=False, mode='auto',
              period=1)]
    model.fit_generator(train_generator,\
                epochs=epochs,steps_per_epoch=steps_per_epoch,validation_data=validation_data,validation_steps=validation_steps, callbacks=callbacks_)

    return model

def save_model(model, job_dir):
    model.save('model.h5')
    # serialize model to JSON
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
        
    # Save model.h5 on to google storage
    with file_io.FileIO('model.h5', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.h5', mode='w') as output_f:
            output_f.write(input_f.read())
        #Sava ArchModel in google
    with file_io.FileIO('model.json', mode='r') as input_f:
        with file_io.FileIO(job_dir + '/model.json', mode='w') as output_f:
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
                        # Load dataset as np array
            if(self.data_type == 1):
                f = BytesIO(file_io.read_file_to_string(self.train_dir +"/" + str(ID[0:2]) + "/" + ID, binary_mode=True))
                X[i, :, :, :] = np.load(f)
                        
                        # Load dataset as image
            elif(self.data_type == 0):
                X[i, :, :, :] = io.imread(self.train_dir +"/" + str(ID[0:2]) + "/" + ID)

            # Store class
            y[i] = int(ID[0:2])

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
        print("File id.txt not Found, or not in pickle.dump type")
    else:
        return len(num_samples)

def discover_input_shape(train_dir = None,data_type=1):
    if train_dir == None:
        raise IOError("File id.txt Variable train_dir not initialized")
    try:
        f = BytesIO(file_io.read_file_to_string(train_dir+"/id.txt", binary_mode=True))
        num_samples = np.load(f)
    except IOError:
        print("File id.txt not Found, or not in pickle.dump type")
    else:

        sample_path = num_samples[0]
        if (data_type == 0):
            from skimage import io
            sample = io.imread(train_dir +"/" + sample_path[0:2] + "/" + sample_path)
        else:
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
        help='Dir to save the predict from vgg',
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
    parser.add_argument(
        '--valid-file',
        help='Location of validation File',
        required=False
    )
    parser.add_argument(
        '--model-file',
        help='Location of weights from model.h5',
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
        print("(W) Batch_size not defined and will be set equal 32")
        batch_size=32
    print("Batch Size:",batch_size)
    #train_generator = get_data(train_dir)
    

    if(job_type == "1"):
        
        if(arguments['epochs']):
            epochs = int(arguments['epochs'])
        else:
            epochs = 10
            input_shape = discover_input_shape(train_dir,int(job_type))
            print("Input Shape:",input_shape)
            model = define_model(input_shape=input_shape)
            train_generator, _ = get_data(train_dir, batch_size=batch_size, input_shape=input_shape, shuffle=True)
            # For validation
            if(arguments['valid_file']):
                valid_dir = arguments['valid_file']
                valid_generator ,_ =  get_data(valid_dir, batch_size=batch_size, input_shape=input_shape, shuffle=True)
                validation_steps = int(np.ceil(discover_num_samples(valid_dir)/batch_size))
            else:
                valid_generator = None
                validation_steps = None

            if(arguments['epochs_steps']):
                steps_per_epoch = int(arguments['epochs_steps'])
            else:
                steps_per_epoch = int(np.ceil(discover_num_samples(train_dir)/batch_size))

            model = train_model(model, train_generator, epochs=epochs,
                        steps_per_epoch=4,validation_data=valid_generator,validation_steps=validation_steps, output_dir=job_dir)
            save_model(model, job_dir)

    elif(job_type == "0"):
        if(arguments['predict_dir']):
                    number_of_samples = discover_num_samples(train_dir)
                # Input_shape from a sample
                    input_shape = discover_input_shape(train_dir,int(job_type))
                    output_predict = arguments['predict_dir']
                    bottleneck_features(train_dir, batch_size=batch_size, number_of_samples=number_of_samples,\
                            input_shape=input_shape, output_dir=output_predict, job_type=int(job_type))
        else:
            print("The predict output dir has not been provided.")

            raise ValueError
    # test / validation
    elif ( job_type == "2"):
        if(arguments['valid_file']):
            valid_dir = arguments['valid_file']
        else:
            print("Valid file not defined")
            raise ValueError
        if(arguments['predict_dir']):
            number_of_samples = discover_num_samples(valid_dir)

            # Input_shape from a sample
            input_shape = discover_input_shape(valid_dir,int(job_type))
            output_predict = arguments['predict_dir']

        if(arguments['model_file']):
                    model = define_model(input_shape=input_shape)
                    model.load_weights(arguments["model_file"])
                    valid_generator,_ = get_data(valid_dir, batch_size=number_of_samples, input_shape=input_shape, shuffle=False, data_type=1)
                    data, target = valid_generator.next() 
                    loss,acc = model.evaluate(x=data, y=target)
                    print("----------Result----------")
                    print(" Testing : loss {} , acc : {}".format(loss,acc))





    else:
        print("Invalid job type.")
        raise ValueError
    

    #np.save("my_weights", model.get_weights)[]


if __name__ == "__main__":
    main()
