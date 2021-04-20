"""
GAN for data augmentation
"""

import os

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *


class DAGAN():

        def __init__(self,
                     n_epochs=750,
                     batch_size=512,
                     input_shape=(128, 128, 1),
                     latent_size=100,
                     n_classes = 3,
                     alpha=0.2,
                     drop_rate=0.5,
                     discriminator_lr=1e-6,
                     generator_lr=1e-4,
                     logging_step=10,
                     r1_gamma=10,
                     out_images_path="outImages",
                     checkpoint_dir="checkpoints"):

                self.n_epochs = n_epochs
                self.batch_size = batch_size
                self.input_shape = input_shape
                self.latent_size = latent_size
                self.n_classes = n_classes
                self.alpha = alpha
                self.drop_rate = drop_rate
                self.discriminator_lr = discriminator_lr
                self.generator_lr = generator_lr
                self.logging_step = logging_step
                self.r1_gamma = r1_gamma
                self.out_images_path = out_images_path
                self.checkpoint_dir = checkpoint_dir

                self.model = self._build_model()

        def create_generator(self):
                leaky = tf.keras.layers.LeakyReLU(self.alpha)

                input_image = Input(self.input_shape)
                input_noise = Input(self.latent_size)

                conv1 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(input_image)
                conv1 = Dropout(0.2)(conv1)

                conv1 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv1)
                pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

                conv2 = Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool1)
                conv2 = Dropout(0.2)(conv2)

                conv2 = Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv2)
                pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

                conv3 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool2)
                conv3 = Dropout(0.2)(conv3)

                conv3 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv3)
                pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

                conv4 = Conv2D(256, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool3)
                conv4 = Dropout(0.2)(conv4)

                conv4 = Conv2D(256, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv4)
                pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

                conv5 = Conv2D(512, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool4)
                conv5 = Dropout(0.2)(conv5)

                conv5 = Conv2D(512, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv5)
                drop5 = Dropout(0.5)(conv5)

                dense1 = Dense(8*8*256, use_bias=False, input_shape=(self.latent_size,))(input_noise)
                dense1 = BatchNormalization()(dense1)
                dense1 = leaky(dense1)
                dense1 = Reshape((8, 8, 256))(dense1)

                # Add the noise here
                decoder_input = concatenate([drop5, dense1])

                up6 = Conv2DTranspose(256, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(decoder_input)
                merge6 = concatenate([conv4,up6], axis = 3)
                conv6 = Conv2D(256, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge6)
                conv6 = Dropout(0.2)(conv6)
                conv6 = Conv2D(512, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv6)

                up7 = Conv2DTranspose(128, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(conv6)
                merge7 = concatenate([conv3,up7], axis = 3)
                conv7 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge7)
                conv7 = Dropout(0.2)(conv7)
                conv7 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv7)

                up8 = Conv2DTranspose(64, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(conv7)
                merge8 = concatenate([conv2,up8], axis = 3)
                conv8 = Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge8)
                conv8 = Dropout(0.2)(conv8)
                conv8 = Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv8)

                up9 = Conv2DTranspose(32, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(conv8)
                merge9 = concatenate([conv1,up9], axis = 3)
                conv9 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge9)
                conv9 = Dropout(0.2)(conv9)
                conv9 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv9)
                conv9 = Dropout(0.2)(conv9)
                conv9 = Conv2D(2, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv9)

                output_image = Conv2D(1, (1, 1), activation='tanh', padding='same')(conv9)

                model = Model(inputs=[input_image, input_noise], outputs=output_image)  

                return model 

        def create_discriminator(self):                
                leaky = tf.keras.layers.LeakyReLU(self.alpha)

                input_image = Input(self.input_shape)

                conv1 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(input_image)
                conv1 = Dropout(0.2)(conv1)

                conv1 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv1)
                pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

                conv2 = Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool1)
                conv2 = Dropout(0.2)(conv2)

                conv2 = Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv2)
                pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

                conv3 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool2)
                conv3 = Dropout(0.2)(conv3)

                conv3 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv3)
                pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

                conv4 = Conv2D(256, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool3)
                conv4 = Dropout(0.2)(conv4)

                conv4 = Conv2D(256, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv4)
                pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

                conv5 = Conv2D(512, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(pool4)
                conv5 = Dropout(0.2)(conv5)

                conv5 = Conv2D(512, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv5)
                drop5 = Dropout(0.5)(conv5)

                gap1 = GlobalAveragePooling2D()(drop5)

                fc1 = Dense(128)(gap1)
                outputs = Dense(1)(fc1)
 
                model = Model(inputs=input_image, outputs=outputs)
                return model

        class _DAGANModel(keras.Model):
                def __init__(self, discriminator, generator, latent_size, num_classes, r1_gamma):
                        super(DAGAN._DAGANModel, self).__init__()
                        self.discriminator = discriminator
                        self.generator = generator
                        self.latent_size = latent_size
                        self.num_classes = num_classes
                        self.r1_gamma=r1_gamma

                        self.loss_tracker_generator = keras.metrics.Mean(name="gen_loss")
                        self.loss_tracker_discriminator = keras.metrics.Mean(name="disc_loss")
                        self.accuracy_real_tracker_discriminator = keras.metrics.Mean(name="disc_acc_real")
                        self.accuracy_fake_tracker_discriminator = keras.metrics.Mean(name="disc_acc_fake")

                def call(self, x):
                    return x

                def compile(self, discriminator_optimizer, generator_optimizer):
                        super(DAGAN._DAGANModel, self).compile()
                        self.generator_optimizer = generator_optimizer
                        self.discriminator_optimizer = discriminator_optimizer

                # Define and element-wise binary cross entropy loss
                def element_wise_cross_entropy_from_logits(self, labels, logits):
                        # Compute the loss element-wise
                        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels = labels, logits = logits)
                        # Compute average to reduce everything to a specific number
                        loss = tf.reduce_mean(losses)
                        return loss

                def generator_loss(self, fake_output):
                        return self.element_wise_cross_entropy_from_logits(tf.ones_like(fake_output), fake_output)

                def discriminator_loss(self, real_output, fake_output):
                        real_loss = self.element_wise_cross_entropy_from_logits(tf.ones_like(real_output), real_output)
                        fake_loss = self.element_wise_cross_entropy_from_logits(tf.zeros_like(fake_output), fake_output)
                        total_loss = real_loss + fake_loss
                        return total_loss

                def train_step(self, data):
                        images, labels = data

                        noise = tf.random.normal([images.shape[0], self.latent_size])
                        fake_labels = np.random.randint(0, self.num_classes, labels.shape[0])

                        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

                                generated_images = self.generator((images, noise), training=True)

                                real_output = self.discriminator((images), training=True)
                                fake_output = self.discriminator((generated_images), training=True)


                                gen_loss = self.generator_loss(fake_output)
                                disc_loss = self.discriminator_loss(real_output, fake_output)

                        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
                        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

                        # Compute metrics
                        self.loss_tracker_generator.update_state(gen_loss)
                        self.loss_tracker_discriminator.update_state(disc_loss)

                        preds_real = tf.round(tf.sigmoid(real_output))
                        accuracy_real = tf.math.reduce_mean(tf.cast(tf.math.equal(preds_real, tf.ones_like(preds_real)), tf.float32))
                        self.accuracy_real_tracker_discriminator.update_state(accuracy_real)

                        preds_fake = tf.round(tf.sigmoid(fake_output))
                        accuracy_fake = tf.math.reduce_mean(tf.cast(tf.math.equal(preds_fake, tf.zeros_like(preds_fake)), tf.float32))
                        self.accuracy_fake_tracker_discriminator.update_state(accuracy_fake)

                        return {'gen_loss': self.loss_tracker_generator.result(), 'disc_loss': self.loss_tracker_discriminator.result(), \
                                'disc_acc_real': self.accuracy_real_tracker_discriminator.result(), 'disc_acc_fake': self.accuracy_fake_tracker_discriminator.result()}

                def test_step(self, data):
                        pass

                @property
                def metrics(self):
                        # We list our `Metric` objects here so that `reset_states()` can be
                        # called automatically at the start of each epoch
                        # or at the start of `evaluate()`.
                        # If you don't implement this property, you have to call
                        # `reset_states()` yourself at the time of your choosing.
                        return [self.loss_tracker_generator, self.loss_tracker_discriminator, self.accuracy_real_tracker_discriminator, self.accuracy_fake_tracker_discriminator]

        def _build_model(self):
                self.generator = self.create_generator()
                self.discriminator = self.create_discriminator()

                model = self._DAGANModel(generator=self.generator, discriminator=self.discriminator, latent_size=self.latent_size, num_classes=self.n_classes, r1_gamma=self.r1_gamma)

                self.generator_optimizer = tf.keras.optimizers.Adam(self.generator_lr, beta_1=0.5, clipvalue=5)
                self.discriminator_optimizer = tf.keras.optimizers.Adam(self.discriminator_lr, beta_1=0.5)

                model.compile(generator_optimizer=self.generator_optimizer, discriminator_optimizer=self.discriminator_optimizer)

                return model

        def generate_latent_points(self):
                # generate points in the latent space
                x_input = np.random.randn(self.latent_size * self.batch_size)
                # reshape into a batch of inputs for the network
                x_input = x_input.reshape(self.batch_size, self.latent_size )
                return [x_input]



        def train_model(self, train_ds, benchmark_noise, benchmark_images, benchmark_labels):
                # set checkpoint directory
                checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
                checkpoint = tf.train.Checkpoint(generator_optimizer=self.model.generator_optimizer,
                                                                                discriminator_optimizer=self.model.discriminator_optimizer,
                                                                                model=self.model)

                # creating dictionaries for history and accuracy for the plots
                history = {}
                history['G_loss'] = []
                history['D_loss'] = []
                history['D_acc_real'] = []
                history['D_acc_fake'] = []

                print("Starting training of the DAGAN model.")

                print("Batches per epoch ", len(train_ds))

                for epoch in range(self.n_epochs+1):
                        # Keep track of the losses at each step
                        epoch_gen_loss = []
                        epoch_disc_loss = []
                        epoch_disc_acc_real = []
                        epoch_disc_acc_fake = []

                        print(f"Starting epoch {epoch} of {self.n_epochs}")

                        for step in range(len(train_ds)):
                                images, labels = next(train_ds)
                                g_loss, d_loss, d_acc_real, d_acc_fake = self.model.train_on_batch(images, labels)

                                epoch_gen_loss.append(g_loss)
                                epoch_disc_loss.append(d_loss)
                                epoch_disc_acc_real.append(d_acc_real)
                                epoch_disc_acc_fake.append(d_acc_fake)

                                if step % self.logging_step == 0:
                                        print(f"\tLosses at step {step}:")
                                        print(f"\t\tGenerator Loss: {g_loss}")
                                        print(f"\t\tDiscriminator Loss: {d_loss}")
                                        print(f"\t\tDisc. Acc Real: {d_acc_real}")
                                        print(f"\t\tDisc. Acc Fake: {d_acc_fake}")


                        if epoch % self.logging_step == 0:
                                generator_images = self.model.generator((benchmark_images, benchmark_noise), training=False)

                                print("Actual images: ")
                                self.plot_fake_figures(benchmark_images, benchmark_labels, 4, epoch, self.out_images_path)

                                print("Generated images: ")
                                self.plot_fake_figures(generator_images, benchmark_labels, 4, epoch, self.out_images_path)

                                checkpoint.save(file_prefix=checkpoint_prefix)

        def plot_losses(self, data, xaxis, yaxis, ylim=0):
                pd.DataFrame(data).plot(figsize=(10,8))
                plt.grid(True)
                plt.xlabel(xaxis)
                plt.ylabel(yaxis)
                if ylim!=0:
                        plt.ylim(0, ylim)
                plt.show()

        @staticmethod
        def plot_fake_figures(x, labels, n, epoch, dir='./'):
 
                labels_dict = {
                      0: "covid-19",
                      1: "normal",
                      2: "viral-pneumonia"
                }
 
                fig = plt.figure(figsize=(6,6))
        
                for i in range(n*n):
                    plt.subplot(n,n,i+1)
                    plt.xticks([])
                    plt.yticks([])
                    img=x[i,:,:,:]
                    # rescale for visualization purposes
                    img = tf.keras.preprocessing.image.array_to_img(img)
                    plt.imshow(img, cmap="gray")
                    plt.xlabel(labels_dict[labels[i]])
                    plt.imshow(img, cmap='gray')
                    plt.savefig('{}/image_at_epoch_{:04d}.png'.format(dir, epoch))
        
                plt.show()