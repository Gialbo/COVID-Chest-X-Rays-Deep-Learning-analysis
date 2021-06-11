import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.layers import *
from keras.models import *
import keras
from keras.optimizers import *

class unetCGAN():
  def __init__(self,
                n_epochs=150,
                batch_size=256,
                input_shape=(128, 128, 1),
                latent_size=100,
                alpha=0.2,
                drop_rate=0.4,
                num_classes = 3,
                discriminator_lr=1e-4,
                generator_lr=1e-4,
                logging_step=10,
                r1_gamma=10,
                out_images_path='/content/drive/MyDrive/BIOINF/checkpoints_GAN/unetCGAN/outImages',
                checkpoint_dir='/content/drive/MyDrive/BIOINF/checkpoints_GAN/unetCGAN',
                use_residual=False):
      
      self.n_epochs = n_epochs
      self.batch_size = batch_size
      self.input_shape = input_shape
      self.latent_size = latent_size
      self.alpha = alpha
      self.num_classes = num_classes
      self.drop_rate = drop_rate
      self.discriminator_lr = discriminator_lr
      self.generator_lr = generator_lr
      self.logging_step = logging_step
      self.r1_gamma = r1_gamma
      self.out_images_path = out_images_path
      self.checkpoint_dir = checkpoint_dir
      self.use_residual = use_residual

      self.model = self._build_model()

  def create_unet_discriminator(self):

    leaky = tf.keras.layers.LeakyReLU(self.alpha)

    input_image = Input(self.input_shape)

    input_label = Input(shape=(1,))

    # Embedding for categorical input
    li = Embedding(self.num_classes, 50)(input_label)

    # Scale embedding to image size
    n_nodes = self.input_shape[0] * self.input_shape[1]
    li = Dense(n_nodes)(li)
   
    # Reshape to image input size
    li = Reshape((self.input_shape[0], self.input_shape[1], 1))(li)
   
    # Concatenate input image and label
    merge = Concatenate()([input_image, li])

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

    # # TODO: Evaluate if correct to shrink to 1 output
    fc1 = Dense(128)(gap1)
    out_enc = Dense(1, name="out_enc")(fc1)

    up6 = Conv2DTranspose(256, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(drop5)
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

    out_dec = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name="out_dec")(conv9)

    model = Model(inputs=[input_image, input_label], outputs=[out_enc, out_dec])  

    return model 

  def create_residual_generator(self):
    leaky = tf.keras.layers.LeakyReLU(self.alpha)

    input_noise = Input(shape=self.latent_size)

    input_label = Input(shape=(1,))

    # Embedding for categorical input
    li = Embedding(self.num_classes, 50)(input_label)
    
    # Match initial image size
    n_nodes = 8 * 8
    li = Dense(n_nodes)(li)
    # reshape to add additional channel
    li = Reshape((8, 8, 1))(li)

    dense1 = Dense(8*8*256, use_bias=False, input_shape=(self.latent_size,))(input_noise)
    dense1 = BatchNormalization()(dense1)
    dense1 = leaky(dense1)
    dense1 = Reshape((8, 8, 256))(dense1)

    merge = Concatenate()([dense1, li])

    up1 = Conv2DTranspose(256, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(merge)
    conv1 = Conv2D(256, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(up1)
    conv1 = Dropout(0.2)(conv1)
    conv1 = Conv2D(512, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv1)

    # Residual connection
    up_res_1 = UpSampling2D(size=(2,2))(up1)

    up2 = Conv2DTranspose(128, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(conv1)
    merge2 = concatenate([up_res_1,up2], axis = 3)
    conv2 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge2)
    conv2 = Dropout(0.2)(conv2)
    conv2 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv2)

    # Residual connection
    up_res_2 = UpSampling2D(size=(2,2))(up2)


    up3 = Conv2DTranspose(64, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(conv2)
    merge3 = concatenate([up_res_2,up3], axis = 3)
    conv3= Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge3)
    conv3= Dropout(0.2)(conv3)
    conv3= Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv3)

    # Residual connection
    up_res_3 = UpSampling2D(size=(2,2))(up3)

    up4 = Conv2DTranspose(32, 2, strides=(2, 2), activation=leaky, padding='same', kernel_initializer='he_normal')(conv3)
    merge4 = concatenate([up_res_3,up4], axis = 3)
    conv4 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge4)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv4)
    conv4 = Dropout(0.2)(conv4)
    conv4 = Conv2D(2, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv4)

    output = Conv2D(1, (1, 1), activation='tanh', padding='same', name="out_dec")(conv4)
    model = Model(inputs=[input_noise, input_label], outputs=[output])
    return model

  def create_generator(self):

    leaky = tf.keras.layers.LeakyReLU(self.alpha)

    input_noise = Input(shape=self.latent_size)

    input_label = Input(shape=(1,))

    # Embedding for categorical input
    li = Embedding(self.num_classes, 50)(input_label)
    
    # Match initial image size
    n_nodes = 8 * 8
    li = Dense(n_nodes)(li)
    # reshape to add additional channel
    li = Reshape((8, 8, 1))(li)

    dense1 = Dense(8*8*256, use_bias=False, input_shape=(self.latent_size,))(input_noise)
    dense1 = BatchNormalization()(dense1)
    dense1 = leaky(dense1)
    dense1 = Reshape((8, 8, 256))(dense1)

    merge = Concatenate()([dense1, li])

    conv1 = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(merge)
    conv1 = BatchNormalization()(conv1)
    conv1 = leaky(conv1)
          
    conv2 = Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(conv1)
    conv2 = BatchNormalization()(conv2)
    conv2 = leaky(conv2)

    conv3 = Conv2DTranspose(32, (5, 5), strides=(2, 2), padding='same', use_bias=False)(conv2)
    conv3 = BatchNormalization()(conv3)
    conv3 = leaky(conv3)

    conv4 = Conv2DTranspose(16, (5, 5), strides=(2, 2), padding='same', use_bias=False)(conv3)
    conv4 = BatchNormalization()(conv4)
    conv4 = leaky(conv4)

    conv5 = Conv2DTranspose(8, (5, 5), strides=(2, 2), padding='same', use_bias=False)(conv4)
    conv5 = BatchNormalization()(conv5)
    conv5 = leaky(conv5)

    output = Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh')(conv4)

    model = Model(inputs=[input_noise, input_label], outputs=[output])
    return model

  class _unetCGANModel(keras.Model):
    def __init__(self, discriminator, generator, latent_size, num_classes, r1_gamma):
        super(unetCGAN._unetCGANModel, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.r1_gamma=r1_gamma

        self.loss_tracker_generator = keras.metrics.Mean(name="gen_loss")
        self.loss_tracker_discriminator = keras.metrics.Mean(name="disc_loss")
        self.loss_true_tracker_discriminator = keras.metrics.Mean(name="disc_loss_real")
        self.loss_fake_tracker_discriminator = keras.metrics.Mean(name="disc_loss_fake")
        self.accuracy_real_tracker_discriminator = keras.metrics.Mean(name="disc_acc_real")
        self.accuracy_fake_tracker_discriminator = keras.metrics.Mean(name="disc_acc_fake")
    
    
    def call(self, x):
      return x

    def compile(self, discriminator_optimizer, generator_optimizer):
      super(unetCGAN._unetCGANModel, self).compile()
      self.generator_optimizer = generator_optimizer
      self.discriminator_optimizer = discriminator_optimizer

    # Define and element-wise binary cross entropy loss
    def element_wise_cross_entropy_from_logits(self, labels, logits):
        # Compute the loss element-wise
        losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits)
        # Compute average to reduce everything to a specific number
        loss = tf.reduce_mean(losses)
        return loss

    def r1_regularization(self, d_logits, true_data):
      """
      Penalizes the gradients of the discriminator on the true data distribution.
      This tecnique is described in: https://arxiv.org/pdf/1801.04406v4.pdf
      This method is adapted from the more general gradient penalization regularization
      fro GANS introduced in this repo: https://github.com/rothk/Stabilizing_GANs
      """
      d = tf.nn.sigmoid(d_logits)

      grad_d_logits = tf.gradients(d_logits, true_data)[0]

      grad_d_logits_norm = tf.norm(tf.reshape(grad_d_logits, [batch_size, -1]), axis=1, keepdims=True)

      disc_regularizer = tf.reduce_mean(grad_d_logits_norm)
      return disc_regularizer

    def generator_loss(self, fake_output):
        return self.element_wise_cross_entropy_from_logits(tf.ones_like(fake_output), fake_output)

    def discriminator_loss(self, real_output, fake_output, data, enc=False):
        real_loss = self.element_wise_cross_entropy_from_logits(tf.ones_like(real_output), real_output)
        fake_loss = self.element_wise_cross_entropy_from_logits(tf.zeros_like(fake_output), fake_output)
        if enc:
          r1_penalty = self.r1_gamma/2 * self.r1_regularization(real_output, data)
          total_loss = real_loss + fake_loss + r1_penalty
        else:
          total_loss = real_loss + fake_loss
        return real_loss, fake_loss, total_loss

    def train_step(self, data):
      
      images, labels = data

      noise = tf.random.normal([images.shape[0], self.latent_size])
      fake_labels = np.random.randint(0, self.num_classes, labels.shape[0])
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = self.generator((noise, fake_labels), training=True)
        
        real_output = self.discriminator((images, labels), training=True)
        fake_output = self.discriminator((generated_images, fake_labels), training=True)

        real_output_enc = real_output[0]
        fake_output_enc = fake_output[0]

        real_output_dec = real_output[1]
        fake_output_dec = fake_output[1]
        
        gen_loss = self.generator_loss(fake_output_enc) + self.generator_loss(fake_output_dec)
        
        disc_loss_true_enc, disc_loss_fake_enc, disc_loss_enc = self.discriminator_loss(real_output_enc, fake_output_enc, images) 
        disc_loss_true_dec, disc_loss_fake_dec, disc_loss_dec = self.discriminator_loss(real_output_dec, fake_output_dec, images) 
        
        disc_loss_true = disc_loss_true_enc + disc_loss_true_dec
        disc_loss_fake = disc_loss_fake_enc + disc_loss_fake_dec
        disc_loss = disc_loss_enc + disc_loss_dec

      gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
      
      self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

      # Compute metrics
      self.loss_tracker_generator.update_state(gen_loss)
      self.loss_tracker_discriminator.update_state(disc_loss)
      self.loss_true_tracker_discriminator.update_state(disc_loss_true)
      self.loss_fake_tracker_discriminator.update_state(disc_loss_fake)

      preds_real = tf.round(tf.sigmoid(real_output))
      accuracy_real = tf.math.reduce_mean(tf.cast(tf.math.equal(preds_real, tf.ones_like(preds_real)), tf.float32))
      self.accuracy_real_tracker_discriminator.update_state(accuracy_real)

      preds_fake = tf.round(tf.sigmoid(fake_output))
      accuracy_fake = tf.math.reduce_mean(tf.cast(tf.math.equal(preds_fake, tf.zeros_like(preds_fake)), tf.float32))
      self.accuracy_fake_tracker_discriminator.update_state(accuracy_fake)

      return {'gen_loss': self.loss_tracker_generator.result(), 'disc_loss': self.loss_tracker_discriminator.result(), \
              'disc_loss_real': self.loss_true_tracker_discriminator.result(), 'disc_loss_fake': self.loss_fake_tracker_discriminator.result(), \
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
        return [self.loss_tracker_generator, self.loss_tracker_discriminator, \
                self.loss_true_tracker_discriminator, self.loss_fake_tracker_discriminator, \
                self.accuracy_real_tracker_discriminator, self.accuracy_fake_tracker_discriminator]

  def _build_model(self):
    if self.use_residual:
      self.generator = self.create_residual_generator()
    else:
      self.generator = self.create_generator()

    self.discriminator = self.create_unet_discriminator()

    model = self._unetCGANModel(generator=self.generator, discriminator=self.discriminator, latent_size=self.latent_size, num_classes=self.num_classes, r1_gamma=self.r1_gamma)

    self.generator_optimizer = tf.keras.optimizers.Adam(self.generator_lr, beta_1=0.5, clipvalue=5)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(self.discriminator_lr, beta_1=0.5)

    model.compile(generator_optimizer=self.generator_optimizer, discriminator_optimizer=self.discriminator_optimizer)

    return model

  def train_model(self, train_ds, benchmark_noise, benchmark_labels):
    # set checkpoint directory
    checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=self.model.generator_optimizer,
                                    discriminator_optimizer=self.model.discriminator_optimizer,
                                    model=self.model)

    # creating dictionaries for history and accuracy for the plots
    self.history = {}
    self.history['G loss'] = []
    self.history['D loss'] = []
    self.history['D loss Real'] = []
    self.history['D loss Fake'] = []
    self.accuracy = {}
    self.accuracy['D accuracy Real'] = []
    self.accuracy['D accuracy Fake'] = []

    print("Starting training of the Unet GAN model.")

    print("Batches per epoch ", len(train_ds))

    for epoch in range(self.n_epochs+1):
      # Keep track of the losses at each step
      epoch_gen_loss = []
      epoch_disc_loss = []
      epoch_disc_loss_true = []
      epoch_disc_loss_fake = []
      epoch_disc_acc_true = []
      epoch_disc_acc_fake = []

      print(f"Starting epoch {epoch} of {self.n_epochs}")

      for step, batch in enumerate(train_ds):
        images, labels = batch
        gen_loss_step, disc_loss_step, disc_loss_true_step, disc_loss_fake_step, disc_acc_true_step, disc_acc_fake_step = self.model.train_on_batch(images, labels)

        epoch_gen_loss.append(gen_loss_step)
        epoch_disc_loss.append(disc_loss_step)
        epoch_disc_loss_true.append(disc_loss_true_step)
        epoch_disc_loss_fake.append(disc_loss_fake_step)
        epoch_disc_acc_true.append(disc_acc_true_step)
        epoch_disc_acc_fake.append(disc_acc_fake_step)

        if step % self.logging_step == 0:
            print(f"\tLosses at step {step}:")
            print(f"\t\tGenerator Loss: {gen_loss_step}")
            print(f"\t\tDiscriminator Loss: {disc_loss_step}")
            print(f"\t\tAccuracy Real: {disc_acc_true_step}")
            print(f"\t\tAccuracy Fake: {disc_acc_fake_step}")

              
      if epoch % self.logging_step == 0:
        generator_images = self.model.generator((benchmark_noise, benchmark_labels), training=False)
                  
        print("Generated images: ")
        self.plot_fake_figures(generator_images, benchmark_labels, 4, epoch, self.out_images_path, "generated")

        print("Decoded maps: ")
        decoded_images = self.model.discriminator((generator_images, benchmark_labels), training=False)[1]
        self.plot_fake_figures(decoded_images, None, 4, epoch, self.out_images_path, "decoded")

      if (epoch % (self.logging_step*5)) == 0:
        checkpoint.save(file_prefix = checkpoint_prefix)

      self.history['G loss'].append(np.array(epoch_gen_loss).mean())
      self.history['D loss'].append(np.array(epoch_disc_loss).mean())
      self.history['D loss Real'].append(np.array(epoch_disc_loss_true).mean())          
      self.history['D loss Fake'].append(np.array(epoch_disc_loss_fake).mean())     
      self.accuracy['D accuracy Real'].append(np.array(epoch_disc_acc_true).mean())     
      self.accuracy['D accuracy Fake'].append(np.array(epoch_disc_acc_fake).mean())

      

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