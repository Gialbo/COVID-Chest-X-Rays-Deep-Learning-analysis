"""
GAN model for COVID-19 class
Bioinformatics, Politecnico di Torino
Authors: Gilberto Manunza, Silvia Giammarinaro
"""

import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import LeakyReLU, Activation
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Reshape, Input
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

class covidGAN():

    def __init__(self,
                  n_epochs=1500,
                  batch_size=128,
                  input_shape=(128, 128, 1),
                  latent_size=100,
                  alpha=0.2,
                  drop_rate=0.4,
                  discriminator_lr=3e-5,
                  generator_lr=2e-4, 
		  logging_step=10,
                  out_images_path='/content/drive/MyDrive/BIOINF/checkpoints_GAN/covidGAN/outImages',
                  checkpoint_dir='/content/drive/MyDrive/BIOINF/checkpoints_GAN/covidGAN'):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.alpha = alpha
        self.drop_rate = drop_rate
        self.discriminator_lr = discriminator_lr
        self.generator_lr = generator_lr
	self.logging_step = logging_step
	self.out_images_path = out_images_path
        self.checkpoint_dir = checkpoint_dir
        

        self._build_model()

    def create_discriminator(self):

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

    def create_generator(self):

        width = self.input_shape[0]
        height = self.input_shape[1]
        channels = self.input_shape[2]
        leaky = tf.keras.layers.LeakyReLU(self.alpha)

        inputs = Input(shape=(self.latent_size,))

        dim1 = width // 4     # because we have 3 transpose conv layers with strides 2,1 -> 
                                # -> we are upsampling by a factor of 4 -> 2*2*1
        dim2 = height // 4
        x = Dense( dim1 * dim2 * 8, activation="relu")(inputs) #20*20*3
        x = BatchNormalization()(x)
        
        x = Reshape((dim1, dim2, 8))(x)

        # now add conv 2D transpose: transposed convoultional or deconvolution
        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation=leaky)(x)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation=leaky)(x)
        x = BatchNormalization()(x)
        # now add final layer
        outputs = Conv2DTranspose(channels, (3, 3), strides=(1, 1), padding="same", activation="tanh")(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model


    def _build_model(self):
        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()
        disc_optimizer = optimizers.Adam(lr=self.discriminator_lr, beta_1=0.5,clipvalue=5)
        self.discriminator.compile(disc_optimizer, "binary_crossentropy", metrics="accuracy")
        self.discriminator.trainable = False
        noise = Input((self.latent_size))
        disc_outputs = self.discriminator(self.generator(noise))
        self.gan = Model(inputs=noise, outputs=disc_outputs)

        self.gan_optimizer = optimizers.Adam(lr=self.generator_lr, beta_1=0.6)#, beta_1=0.5)
        self.gan.compile(loss="binary_crossentropy", optimizer= self.gan_optimizer)
        print("GAN model created")
        #self.gan.summary()

    def generate_latent_points(self):
	# generate points in the latent space
        x_input = np.random.randn(self.latent_size * self.batch_size)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(self.batch_size, self.latent_size )
        return x_input

    

    def train_model(self, train, training_size, benchmarkNoise, checkpoint_prefix):

        # creating dictionaries for history and accuracy for the plots
        self.history = {}
        self.history['G_loss'] = []
        self.history['D_loss_true'] = []
        self.history['D_loss_fake'] = []
        self.accuracy = {}
        self.accuracy['Acc_true'] = []
        self.accuracy['Acc_fake'] = []

        batchesPerEpoch = int(training_size / self.batch_size)
        print("Batches per epoch ", batchesPerEpoch)
            

        for epoch in range(self.n_epochs):
            print("Starting epoch ", epoch)
            
            for b in (range(batchesPerEpoch)):
            
                if b == batchesPerEpoch / 2:
                    print("Half epoch done")

                # GENERATE NOISE
                noise =  self.generate_latent_points()

                # now train the discriminator to differentiate between true and fake images

                # DISCRIMINATOR TRAINING ON REAL IMAGES
                trueImages, _ = next(train)
                # true images: label = 1
                y = np.ones((trueImages.shape[0]))
                discLoss, discAccTrue = self.discriminator.train_on_batch(trueImages, y)

                # GENERATOR GENERATING ON FAKE IMAGES
                genImages=self.generator.predict(noise)
                # fake images: label = 0
                y = np.zeros((self.batch_size))

                # DISCRIMINATOR TRAINING ON FAKE IMAGES
                discLoss, discAccFalse = self.discriminator.train_on_batch(genImages, y)

                # GENERATOR TRAINING ON FAKE IMAGES (label 1 for fake images in this case)
                noise = np.random.uniform(-1, 1, size=(self.batch_size,self.latent_size))
                fake_labels = [1] * self.batch_size
                fake_labels = np.reshape(fake_labels, (-1,))
                ganLoss = self.gan.train_on_batch(noise, fake_labels)
		
		if step % self.logging_step == 0:
		  print(f"\tLosses at step {step}:")
		  print(f"\t\tGenerator Loss: {ganLoss}")
		  print(f"\t\tDiscriminator Loss: {discLoss}")
		  print(f"\t\tAccuracy True: {discAccTrue}")
		  print(f"\t\tAccuracy False: {discAccFalse}")
            
            # at the end of each epoch 
            #print("epoch " + str(epoch) + ": discriminator loss " + str(discLoss)+  " - generator loss " + str(ganLoss))
            #print("Accuracy true: " + str(discAccTrue) + " accuracy false: " + str(discAccFalse))

	    
            self.history['D_loss_true'].append(discLoss)          
            self.accuracy['Acc_true'].append(discAccTrue)
	    self.history['D_loss_fake'].append(discLoss)          
            self.accuracy['Acc_fake'].append(discAccFalse)
	    self.history['G_loss'].append(ganLoss)
	
            if (epoch % self.logging_step) == 0:
	        images = self.generator.predict(benchmarkNoise)
                self.plot_fake_figures(images,4, epoch)

            if (epoch % self.logging_step*5) == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
		
	return history

    def plot_losses(self, data, xaxis, yaxis, ylim=0):
      pd.DataFrame(data).plot(figsize=(10,8))
      plt.grid(True)
      plt.xlabel(xaxis)
      plt.ylabel(yaxis)
      if ylim!=0:
        plt.ylim(0, ylim)
      plt.show()
        
    @staticmethod
    def plot_fake_figures(x, n, epoch, dir='/content/drive/MyDrive/BIOINF/checkpoints_GAN/covidGAN/outImages'):
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
            plt.imshow(img.reshape(128, 128), cmap='gray')
        plt.savefig('{}/image_at_epoch_{:04d}.png'.format(dir, epoch))
    
        plt.show()

    
