#Multiple Models in Keras 

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation

#flattened multiple fully-connected-layers
def fc_NN_k(input_dim, hidden_units, output_dim):
    print(input_dim, hidden_units, output_dim)
    model = Sequential()
    model.add(Dense(hidden_units[0], activation='relu', input_shape=(input_dim,), kernel_initializer='he_normal') )
    for i in range(1,len(hidden_units),1):
        model.add(Dense(hidden_units[i], activation='relu', kernel_initializer='he_normal') )
    model.add(Dense(output_dim, kernel_initializer='he_normal'))
    return model