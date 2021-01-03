from keras.models import *
from keras.layers import *
from keras.optimizers import *

def generator_model(input_size=(100, )):
    inputs = Input(input_size)

    dense1 = Dense(8*8*256, use_bias=False, input_shape=(100,))(inputs)
    dense1 = BatchNormalization()(dense1)
    dense1 = LeakyReLU(dense1)
    dense1 = Reshape((8, 8, 256))
    assert dense1 == (None, 8, 8, 256) # Note: None is the batch size

    conv1 = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(dense1)
    assert conv1 == (None, 16, 16, 128)
    conv1 = BatchNormalization()(conv1)
    conv1 = LeakyReLU()(conv1)
    
    conv2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(conv1)
    assert conv2 == (None, 32, 32, 64)
    conv2 = BatchNormalization()(conv2)
    conv2 = LeakyReLU()(conv2)

    conv3 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(conv2)
    assert conv2 == (None, 64, 64, 64)
    conv3 = BatchNormalization()(conv3)
    conv3 = LeakyReLU()(conv3)

    output = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(conv3)
    assert  output == (None, 128, 128, 1)

    model = Model(inputs, output)
    return model
    
    