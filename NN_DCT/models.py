#Multiple Models in Keras 

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

def layered_NN_1(input_dim, units, output_dim):
    print(input_dim, units, output_dim)
    model = Sequential()
    model.add(Dense(units, activation='relu', input_shape=(input_dim,), kernel_initializer='he_normal') )
    model.add(Dense(output_dim, kernel_initializer='he_normal'))

    return model

