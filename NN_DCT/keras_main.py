import keras
import os
from scipy.io import loadmat
import numpy as np
from models import layered_NN_1

flatten=True
data=loadmat('data.mat')

x_train = data['x_train']
x_test  = data['x_test']
y_train = data['y_train']
y_test  = data['y_test']

train_shape = np.shape(x_train)
test_shape = np.shape(x_test)

if flatten == True:
    x_train = np.reshape(x_train, [train_shape[-1], -1])
    x_test  = np.reshape(x_test, [test_shape[-1], -1])
    y_train = np.reshape(y_train, [train_shape[-1], -1])
    y_test  = np.reshape(y_test, [test_shape[-1], -1])


input_dims = np.shape(x_train)[1] 

hidden_units=[1000]
model = layered_NN_1(input_dims, hidden_units[0], input_dims)


model.compile(loss='mse',
                  optimizer=keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True),
                  metrics = ['accuracy'])

print(model.summary())


model.fit(x=x_train, y=y_train, epochs=5, batch_size=32, validation_split=0.1, shuffle=True)
loss_and_metrics = model.evaluate(x_test, y_test, batch_size=128)

