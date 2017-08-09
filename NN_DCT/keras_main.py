import keras
import os
from scipy.io import loadmat
import numpy as np
from models import fc_NN_k
import tensorflow as tf
from utils import load_dataset

# gpu options
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
set_session(tf.Session(config=config))

#set parameters (TODO add using argparse)
flatten = True
num_epochs = 100
batch_size = 32
validation_split = 0.1
dataset = 'mnist'

#load dataset
(x_train, y_train), (x_test, y_test) = load_dataset(dataset, flatten)
#preprocess train/test dataset for neural network
#no-need in case of mnist


# only considering flattened models (only fc layers) for now
input_dims = np.shape(x_train)[1] 
hidden_units = [1000]


#define the model
model = fc_NN_k(input_dims, hidden_units, input_dims)
model.compile(loss='mse',
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
                  metrics = ['accuracy'])
print(model.summary())

#train and test the model
model.fit(x=x_train, y=y_train, epochs=num_epochs  , batch_size=batch_size, validation_split=validation_split, shuffle=True)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=batch_size)

#save the model for further analysis
model.save('temp.h5')

