"""
GAN for data augmentation
"""

import os
import random

import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from tensorflow.python.ops.gen_batch_ops import batch


class DAGAN():

        def __init__(self,
                     n_epochs=750,
                     batch_size=512,
                     input_shape=(128, 128, 1),
                     latent_size=100,
                     n_classes = 3,
                     alpha=0.2,
                     drop_rate=0.5,
                     discriminator_lr=2e-4,
                     generator_lr=2e-4,
                     logging_step=10,
                     r1_gamma=20,
                     clip_value=0.01,
                     num_critic=5,
                     gp_weight=10,
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
                self.clip_value = clip_value
                self.num_critic = num_critic
                self.gp_weight = gp_weight
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

                real_input = Input(self.input_shape)
                conditional_input = Input(self.input_shape)

                merge = concatenate([real_input, conditional_input])

                conv1 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge)
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
 
                model = Model(inputs=[real_input, conditional_input], outputs=outputs)
                return model

        class _DAGANModel(keras.Model):
                def __init__(self, discriminator, generator, latent_size, inner_batch_size, num_classes, r1_gamma, clip_value, num_critic, gp_weight):
                        super(DAGAN._DAGANModel, self).__init__()
                        self.discriminator = discriminator
                        self.generator = generator
                        self.latent_size = latent_size
                        self.num_classes = num_classes
                        self.r1_gamma = r1_gamma
                        self.clip_value = clip_value
                        self.num_critic = num_critic
                        self.gp_weight = gp_weight
                        # This batch size is different of those of data as we consider self.num_critics half batches size each train step for the discriminator
                        self.inner_batch_size = inner_batch_size
 
                        self.loss_tracker_generator = keras.metrics.Mean(name="gen_loss")
                        self.loss_tracker_real_discriminator = keras.metrics.Mean(name="disc_loss_real")
                        self.loss_tracker_fake_discriminator = keras.metrics.Mean(name="disc_loss_fake")
                        self.accuracy_real_tracker_discriminator = keras.metrics.Mean(name="disc_acc_real")
                        self.accuracy_fake_tracker_discriminator = keras.metrics.Mean(name="disc_acc_fake")
 
                def call(self, x):
                    return x
 
                def compile(self, discriminator_optimizer, generator_optimizer):
                        super(DAGAN._DAGANModel, self).compile()
                        self.generator_optimizer = generator_optimizer
                        self.discriminator_optimizer = discriminator_optimizer

                def gradient_penalty(self, batch_size, real_images_i, real_images_j, fake_images):
                        """ 
                        Calculates the gradient penalty.
                        This loss is calculated on an interpolated image
                        and added to the discriminator loss.
                        """
                        # Get the interpolated image
                        alpha = tf.random.normal([batch_size, 1, 1, 1], 0.0, 1.0)
                        diff = fake_images - real_images_j
                        interpolated = real_images_j + alpha * diff

                        with tf.GradientTape() as gp_tape:
                            gp_tape.watch(interpolated)
                            # 1. Get the discriminator output for this interpolated image.
                            pred = self.discriminator((real_images_i, interpolated), training=True)

                        # 2. Calculate the gradients w.r.t to this interpolated image.
                        grads = gp_tape.gradient(pred, [interpolated])[0]
                        # 3. Calculate the norm of the gradients.
                        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
                        gp = tf.reduce_mean((norm - 1.0) ** 2)
                        return gp
 
                def generator_loss(self, fake_img):
                        return -tf.reduce_mean(fake_img)
 
                def discriminator_loss(self, real_img, fake_img):
                        real_loss = tf.reduce_mean(real_img)
                        fake_loss = tf.reduce_mean(fake_img)
                        return real_loss, fake_loss

                def split_half_batches(self, images, num_critic, batch_size):
                        """
                        Split a dataset in num_critics partitions of size batch_size / 2
                        """

                        half_batch_images_subsets = []

                        for i in range(num_critic):
                                start_idx = int(i * batch_size / 2)
                                end_idx = int(start_idx + (batch_size / 2))
                                half_batch_images_subsets.append(images[start_idx:end_idx])

                        return half_batch_images_subsets
                
                def train_step(self, data):
                        total_images = data

                        # Get num_critics subsets of half batches of the true samples
                        half_batch_images_subsets = self.split_half_batches(total_images[0], self.num_critic+2, self.inner_batch_size)

                        # Update_critic
                        for i in range(self.num_critic):
                                
                                i_batch_index = random.randint(0, self.num_critic + 2 -1)
                                half_batch_images_i = half_batch_images_subsets[i_batch_index]
                                j_batch_index = i_batch_index
                                while j_batch_index == i_batch_index:
                                        j_batch_index = random.randint(0, self.num_critic + 2 - 1)
 
                                half_batch_images_j = half_batch_images_subsets[j_batch_index]

                                half_batch_noise = tf.random.normal([half_batch_images_i.shape[0], self.latent_size])

                                # half_batch_fake_labels = np.random.randint(0, self.num_classes, half_batch_labels_subsets[i].shape[0])
                                with tf.GradientTape() as disc_tape:  
                                        half_batch_generated_images = self.generator((half_batch_images_i, half_batch_noise), training=True)
  
                                        half_batch_real_output = self.discriminator((half_batch_images_i, half_batch_images_j), training=True)
                                        half_batch_fake_output = self.discriminator((half_batch_images_i, half_batch_generated_images), training=True)

                                        real_loss, fake_loss = self.discriminator_loss(half_batch_real_output, half_batch_fake_output)

                                        gp = self.gradient_penalty(self.inner_batch_size // 2, half_batch_images_i, half_batch_images_j, half_batch_generated_images)

                                        disc_loss = fake_loss - real_loss + gp * self.gp_weight
  
                        
                                gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

                                # Clip the gradients of the discriminator
                                # gradients_of_discriminator = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value)) for grad in gradients_of_discriminator]
                                self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))
                        

                        full_batch_noise = tf.random.normal([self.inner_batch_size, self.latent_size])
                        i_batch_index = random.randint(0, self.num_critic + 2 -1)
                        half_batch_images_i = half_batch_images_subsets[i_batch_index]
                        j_batch_index = i_batch_index
                        while j_batch_index == i_batch_index:
                                j_batch_index = random.randint(0, self.num_critic + 2 - 1)
                        half_batch_images_j = half_batch_images_subsets[j_batch_index]

                        full_batch_images = tf.concat([half_batch_images_i, half_batch_images_j], 0)

                        with tf.GradientTape() as gen_tape:
                                full_batch_generated_images = self.generator((full_batch_images, full_batch_noise), training=True)
                                full_batch_fake_output = self.discriminator((full_batch_images, full_batch_generated_images), training=True)

                                gen_loss = self.generator_loss(full_batch_fake_output)
                        
                        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)

                        # gradients_of_generator = [(tf.clip_by_value(grad, -self.clip_value, self.clip_value)) for grad in gradients_of_generator]
                        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
                        
 
                        # Compute metrics
                        self.loss_tracker_generator.update_state(gen_loss)
                        self.loss_tracker_real_discriminator.update_state(real_loss)
                        self.loss_tracker_fake_discriminator.update_state(fake_loss)
                        
                        preds_real = tf.where(tf.less(half_batch_real_output, 0.5), half_batch_real_output, tf.ones_like(half_batch_real_output))
                        preds_real = tf.where(tf.greater(half_batch_real_output, 0.5), preds_real, tf.zeros_like(preds_real))
                        accuracy_real = tf.math.reduce_mean(tf.cast(tf.math.equal(preds_real, tf.ones_like(preds_real)), tf.float32))
                        self.accuracy_real_tracker_discriminator.update_state(accuracy_real)

                        preds_fake = tf.where(tf.less(half_batch_fake_output, 0.5), half_batch_fake_output, tf.ones_like(half_batch_fake_output))
                        preds_fake = tf.where(tf.greater(half_batch_fake_output, 0.5), preds_fake, tf.zeros_like(preds_fake))
                        accuracy_fake = tf.math.reduce_mean(tf.cast(tf.math.equal(preds_fake, tf.zeros_like(preds_fake)), tf.float32))
                        self.accuracy_fake_tracker_discriminator.update_state(accuracy_fake)

                        return {'gen_loss': self.loss_tracker_generator.result(), 'disc_loss_real': self.loss_tracker_real_discriminator.result(), 'disc_loss_fake': self.loss_tracker_fake_discriminator.result(),\
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
                        return [self.loss_tracker_generator, self.loss_tracker_real_discriminator, self.loss_tracker_fake_discriminator, self.accuracy_real_tracker_discriminator, self.accuracy_fake_tracker_discriminator]

        def _build_model(self):
                self.generator = self.create_generator()
                self.discriminator = self.create_discriminator()

                model = self._DAGANModel(generator=self.generator, discriminator=self.discriminator, latent_size=self.latent_size, inner_batch_size=int(self.batch_size/(self.num_critic+2)*2),
                                          num_classes=self.n_classes, r1_gamma=self.r1_gamma, clip_value=self.clip_value, num_critic=self.num_critic, gp_weight=self.gp_weight)

                self.generator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)
                self.discriminator_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5, beta_2=0.9)

                model.compile(generator_optimizer=self.generator_optimizer, discriminator_optimizer=self.discriminator_optimizer)

                return model

        def generate_latent_points(self, size):
                x_input = tf.random.normal([size, self.latent_size])
                return x_input



        def train_model(self, train_datasets, benchmark_noise, benchmark_images, benchmark_labels):
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
                
                # Train datasets is a list containing a dataset for each class from which fetch data from
                batches_per_epocs = np.sum([len(train_ds) for train_ds in train_datasets])
                print("Batches per epoch ", batches_per_epocs)

                for epoch in range(self.n_epochs+1):
                        # Keep track of the losses at each step
                        epoch_gen_loss = []
                        epoch_disc_loss_real = []
                        epoch_disc_loss_fake = []
                        epoch_disc_acc_real = []
                        epoch_disc_acc_fake = []

                        print(f"Starting epoch {epoch} of {self.n_epochs}")

                        for step in range(batches_per_epocs):

                                # Extract a batch from a class for this epoch
                                # Each step I have an equal prob of choosing a class
                                chosen_dataset = random.randint(0, len(train_datasets) - 1)
                                train_ds = train_datasets[chosen_dataset]


                                images = next(train_ds.__iter__())
                                g_loss, d_loss_real, d_loss_fake, d_acc_real, d_acc_fake = self.model.train_on_batch(images)

                                epoch_gen_loss.append(g_loss)
                                epoch_disc_loss_real.append(d_loss_real)
                                epoch_disc_loss_fake.append(d_loss_fake)
                                epoch_disc_acc_real.append(d_acc_real)
                                epoch_disc_acc_fake.append(d_acc_fake)

                                if step % self.logging_step == 0:
                                        print(f"\tLosses at step {step}:")
                                        print(f"\t\tGenerator Loss: {g_loss}")
                                        print(f"\t\tDiscriminator Loss Real: {d_loss_real}")
                                        print(f"\t\tDiscriminator Loss Fake: {d_loss_fake}")
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
