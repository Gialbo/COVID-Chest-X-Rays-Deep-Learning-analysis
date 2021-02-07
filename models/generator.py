from keras.models import *
from keras.layers import *
from keras.optimizers import *

def Generator(input_size=(100, )):
    inputs = Input(input_size)

    dense1 = Dense(8*8*512, use_bias=False, input_shape=(100,))(inputs)
    dense1 = BatchNormalization()(dense1)
    dense1 = LeakyReLU()(dense1)
    dense1 = Reshape((8, 8, 512))(dense1)

    conv1 = Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False)(dense1)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)
    
    conv2 = Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)

    conv3 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)

    conv4 = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = LeakyReLU()(conv4)

    conv5 = Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False)(conv3)
    conv5 = BatchNormalization()(conv4)
    conv5 = LeakyReLU()(conv4)

    output = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(conv5)

    model = Model(inputs, output)
    return model
    
    