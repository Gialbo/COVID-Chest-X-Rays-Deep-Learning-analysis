import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import LeakyReLU, Activation
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Reshape, Input
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

class rawGAN:

    def __init__(self,
                n_epochs,
                batch_size,
                input_shape,
                latent_size,
                alpha,
                drop_rate,
                discriminator_lr,
                generator_lr):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.apha = alpha
        self.drop_rate = drop_rate
        self.discriminator_lr = discriminator_lr
        self.generator_lr = generator_lr

        self._build_model

    def create_discriminator():

        leaky = tf.keras.layers.LeakyReLU(self.alpha)

        inputs = Input(shape=self.input_shape)
        x = Conv2D(32, (5,5), activation=leaky)(inputs)
        x = MaxPooling2D(strides=2)(x)
        x = Conv2D(64, (5,5), activation=leaky)(x)
        x = MaxPooling2D(strides=2)(x)
        x = Conv2D(128, (5,5), activation=leaky)(x)
        x = MaxPooling2D(strides=2)(x)
        x = Flatten()(x)
        x = Dense(128, activation=leaky)(x)
        x = Dropout(self.drop_rate)(x)
        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def create_generator():

        width = self.input_shape[0]
        height = self.input_shape[1]
        channels = self.input_shape[2]
        leaky = tf.keras.layers.LeakyReLU(self.alpha)

        inputs = Input(shape=(self.latent_size,))

        dim1 = width // 4     # because we have 3 transpose conv layers with strides 2,1 -> 
                                # -> we are upsampling by a factor of 4 -> 2*2*1
        dim2 = height // 4
        x = Dense( dim1 * dim2 * n1, activation="relu")(inputs) #20*20*3
        x = BatchNormalization()(x)
        
        x = Reshape((dim1, dim2, n1))(x)

        # now add conv 2D transpose: transposed convoultional or deconvolution
        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation=leaky)(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation=leaky)(x)
        x = BatchNormalization()(x)
        # now add final layer
        outputs = Conv2DTranspose(channels, (3, 3), strides=(1, 1), padding="same", activation="tanh")(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model


    def _build_model():
        discriminator = create_discriminator(self.input_shape)
        generator = create_generator(self.input_shape)
        disc_optimizer = optimizers.Adam(lr=self.discriminator_lr, beta_1=0.5,clipvalue=5)
        discriminator.compile(disc_optimizer, "binary_crossentropy", metrics="accuracy")
        discriminator.trainable = False
        noise = Input((self.latent_size))
        disc_outputs = discriminator(generator(noise))
        self.gan = Model(inputs=noise, outputs=disc_outputs)

        gan_optimizer = optimizers.Adam(lr=self.generator_lr, beta_1=0.5)
        self.gan.compile(loss="binary_crossentropy", optimizer= gan_optimizer)
        print("GAN model created")
        #self.gan.summary()

    def generate_latent_points():
	    # generate points in the latent space
        x_input = np.random.randn(self.latent_dim * self.batch_size)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(self.batch_size, self.latent_dim )
        return 
        
    def plot_fake_figures(x, n, dir='/content/drive/MyDrive/BIOINF/images_GAN/one-class'):
        fig = plt.figure(figsize=(6,6))
        for i in range(n*n):
            plt.subplot(n,n,i+1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            img=x[i,:,:,:]
            # rescale for visualization purposes
            #img = np.repeat(img, 3, axis=-1)
            img = ((img*127.5) + 127.5).astype("uint8")
            plt.imshow(img)
        plt.savefig('{}/image_at_epoch_{:04d}.png'.format(dir, epoch))
    
        plt.show()

    #def train_model(model):
