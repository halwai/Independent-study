import cv2
import keras
from keras.datasets import mnist, cifar10
import numpy as np

def img_2_dct(images, input_size, rgb=True):
    final_images = np.zeros((input_size[0], input_size[1], input_size[2]))
    output_images = np.zeros((input_size[0], input_size[1], input_size[2]))

    for i in range(len(images)):
        if rgb:
            final_images[i,:,:] = cv2.cvtColor(images[i,:,:],cv2.COLOR_RGB2GRAY)/255.0
        else:
            final_images[i,:,:] = images[i,:,:]/255.0
        output_images[i,:,:] = cv2.dct(final_images[i,:,:])

    return (final_images, output_images)

def load_dataset(data_string, flatten):
    if data_string =='mnist':
        (x_train_temp, _ ), (x_test_temp, _ ) = mnist.load_data()

        train_shape = np.shape(x_train_temp)
        test_shape = np.shape(x_test_temp)

        #load the final mnist images inputs and ouputs(dcts)
        (x_train, y_train) = img_2_dct(x_train_temp, train_shape, rgb= len(train_shape)>3)
        (x_test, y_test) = img_2_dct(x_test_temp, test_shape, rgb= len(test_shape)>3)

        if flatten == True:
            x_train = np.reshape(x_train, [train_shape[0], -1])
            y_train = np.reshape(y_train, [train_shape[0], -1])
            x_test  = np.reshape(x_test, [test_shape[0], -1])
            y_test  = np.reshape(y_test, [test_shape[0], -1])


    elif data_string =='cifar10':
        (x_train_temp, _ ), (y_train_temp, _) = cifar10.load_data()
        
        train_shape = np.shape(x_train_temp)
        test_shape = np.shape(x_test_temp)

        #load the final cifar10 images inputs and ouputs(dcts)
        (x_train, y_train) = img_2_dct(x_train_temp, train_shape, rgb= len(train_shape)>3)
        (x_test, y_test) = img_2_dct(x_test_temp, test_shape, rgb= len(test_shape)>3)

        if flatten == True:
            x_train = np.reshape(x_train, [train_shape[0], -1])
            y_train = np.reshape(y_train, [train_shape[0], -1])
            x_test  = np.reshape(x_test, [test_shape[0], -1])
            y_test  = np.reshape(y_test, [test_shape[0], -1])

    else:
        print(data_string)
        raise ValueError('Requested dataset is not available')


    return ((x_train, y_train), (x_test, y_test))


