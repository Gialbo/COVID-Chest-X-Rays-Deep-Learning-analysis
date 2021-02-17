import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv2DTranspose, Conv2D, MaxPooling2D, BatchNormalization, SpatialDropout2D
from tensorflow.keras.layers import LeakyReLU, Activation
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.layers import Reshape, Input
from tensorflow.keras import optimizers
import matplotlib.pyplot as plt

class cGAN():

    def __init__(self,
                  n_epochs=400,
                  batch_size=128,
                  input_shape=(128, 128, 3),
                  latent_size=100,
                  n_classes = 3,
                  alpha=0.2,
                  drop_rate=0.4,
                  discriminator_lr=5e-5,
                  generator_lr=1e-4):

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.latent_size = latent_size
        self.n_classes = n_classes
        self.alpha = alpha
        self.drop_rate = drop_rate
        self.discriminator_lr = discriminator_lr
        self.generator_lr = generator_lr
        

        self._build_model()

    def create_discriminator(self):

        # label input
        in_label = Input(shape=(1,))
        # scale up to image dimensions with linear activation
        n_nodes = self.input_shape[0] * self.input_shape[1]
        li = Dense(n_nodes)(in_label)
        # reshape to additional channel
        li = Reshape((self.input_shape[0], self.input_shape[1], 1))(li)
        # image input
        in_image = Input(shape=self.input_shape)
        # concat label as a channel
        merge = Concatenate()([in_image, li])

        leaky = tf.keras.layers.LeakyReLU(self.alpha)
        x = Conv2D(32, (5,5), activation=leaky)(merge)
        x = MaxPooling2D(strides=2)(x)
        x = Conv2D(64, (5,5), activation=leaky)(x)
        x = MaxPooling2D(strides=2)(x)
        x = Conv2D(128, (5,5), activation=leaky)(x)
        x = MaxPooling2D(strides=2)(x)
        x = Flatten()(x)
        x = Dense(128, activation=leaky)(x)
        x = Dropout(self.drop_rate)(x)
        outputs = Dense(1, activation="sigmoid")(x)

        model = Model(inputs=[in_image, in_label], outputs=outputs)
        return model

    def create_generator(self):

        width = self.input_shape[0]
        height = self.input_shape[1]
        channels = self.input_shape[2]
        leaky = tf.keras.layers.LeakyReLU(self.alpha)


        dim1 = width // 4     # because we have 3 transpose conv layers with strides 2,1 -> 
                                # -> we are upsampling by a factor of 4 -> 2*2*1
        dim2 = height // 4
        
        in_label = Input(shape=(1,))
        n_nodes = dim1 * dim2
        li = Dense(n_nodes)(in_label)
        li = Reshape((dim1, dim2, 1))(li)
        in_lat = Input(shape=(self.latent_size,))
        
        x = Dense( dim1 * dim2 * 16, activation="relu")(in_lat) #20*20*3
        x = BatchNormalization()(x)
        
        x = Reshape((dim1, dim2, 16))(x)

        merge = Concatenate()([x, li])

        # now add conv 2D transpose: transposed convoultional or deconvolution
        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same", activation=leaky)(merge)
        x = BatchNormalization()(x)
        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same", activation=leaky)(x)
        x = BatchNormalization()(x)
        # now add final layer
        outputs = Conv2DTranspose(channels, (3, 3), strides=(1, 1), padding="same", activation="tanh")(x)

        model = Model(inputs=[in_lat, in_label], outputs=outputs)
        return model


    def _build_model(self):
        self.discriminator = self.create_discriminator()
        self.generator = self.create_generator()

        disc_optimizer = optimizers.Adam(lr=self.discriminator_lr, beta_1=0.5,clipvalue=5)
        self.discriminator.compile(disc_optimizer, "binary_crossentropy", metrics="accuracy")

        self.discriminator.trainable = False
        gen_noise, gen_label = self.generator.input
        gen_output = self.generator.output
        gan_output = self.discriminator([gen_output, gen_label])
        self.gan = Model(inputs=[gen_noise, gen_label], outputs=gan_output)
        self.gan_optimizer = optimizers.Adam(lr=self.generator_lr, beta_1=0.5)
        self.gan.compile(loss="binary_crossentropy", optimizer= self.gan_optimizer)
        print("GAN model created")
        #self.gan.summary()

    def generate_latent_points(self):
	      # generate points in the latent space
        x_input = np.random.randn(self.latent_size * self.batch_size)
        # reshape into a batch of inputs for the network
        x_input = x_input.reshape(self.batch_size, self.latent_size )
        # generate labels
        labels = np.random.randint(0, self.n_classes, batch_size)
        return [x_input, labels]

    

    def train_model(self, train, training_size, benchmarkImages, benchmarkLabels, checkpoint_prefix):

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
                noise_img, noise_labels  =  self.generate_latent_points()

                # now train the discriminator to differentiate between true and fake images

                # DISCRIMINATOR TRAINING ON REAL IMAGES
                trueImages, trueLabels = next(train)
                # true images: label = 1
                y = np.ones((trueImages.shape[0], 1))
                discLoss, discAcc = self.discriminator.train_on_batch([trueImages, trueLabels], y)
                self.history['D_loss_true'].append(discLoss)          
                self.accuracy['Acc_true'].append(discAcc)

                # GENERATOR GENERATING ON FAKE IMAGES AND LABELS
                genImages =self.generator.predict([noise_img, noise_labels])
                # fake images: label = 0
                y = np.zeros((self.batch_size, 1))

                # DISCRIMINATOR TRAINING ON FAKE IMAGES
                discLoss, discAcc = self.discriminator.train_on_batch([genImages, noise_labels], y)
                self.history['D_loss_fake'].append(discLoss)          
                self.accuracy['Acc_fake'].append(discAcc)

                # GENERATOR TRAINING ON FAKE IMAGES (label 1 for fake images in this case)

                # GENERATE NOISE
                noise_img, noise_labels  =  self.generate_latent_points()

                fake_labels = np.zeros((self.batch_size, 1))
                ganLoss = self.gan.train_on_batch([noise_img, noise_labels], fake_labels)
                self.history['G_loss'].append(ganLoss)
            
            # at the end of each epoch 
            print("epoch " + str(epoch) + ": discriminator loss " + str(discLoss)+  " ( "  + str(discAcc) + " ) - generator loss " + str(ganLoss))

            images = self.generator.predict([benchmarkImages, benchmarkLabels])
            if (epoch % 10) == 0:
                self.plot_fake_figures(images,3, epoch)

            if (epoch % 50) == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
        
    @staticmethod
    def plot_fake_figures(x, n, epoch, dir='/content/drive/MyDrive/BIOINF/images_GAN/one-class'):
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
            plt.imshow(img, cmap="gray_r")
        plt.savefig('{}/image_at_epoch_{:04d}.png'.format(dir, epoch))
    
        plt.show()

    def plot_losses(history):
        pd.DataFrame(history).plot(figsize=(10,8))
        plt.grid(True)
        plt.show()