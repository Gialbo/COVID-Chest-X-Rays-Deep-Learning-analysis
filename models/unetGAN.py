import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from keras.layers import *
from keras.models import *
import keras
from keras.optimizers import *

class unetGAN():
  def __init__(self,
                  n_epochs=1500,
                  batch_size=256,
                  input_shape=(128, 128, 1),
                  latent_size=100,
                  alpha=0.2,
                  drop_rate=0.4,
                  discriminator_lr=5e-5,
                  generator_lr=1e-4,
                  logging_step=10,
                  out_images_path="outImages",
                  checkpoint_dir="checkpoints"):
        
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
        self.checkpoint_dir = "checkpoints"

        self.model = self._build_model()

  def create_unet_discriminator(self):

    leaky = tf.keras.layers.LeakyReLU(self.alpha)

    inputs = Input(self.input_shape)

    conv1 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(inputs)
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

    # TODO: Evaluate if correct to shrink to 1 output
    # fc1 = Dense(128)(gap1)
    # out_enc = Dense(1, name="out_enc")(fc1)
    out_enc = gap1

    up6 = Conv2D(256, 2, activation=leaky, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(drop5))
    merge6 = concatenate([conv4,up6], axis = 3)
    conv6 = Conv2D(256, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Dropout(0.2)(conv6)
    conv6 = Conv2D(512, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(128, 2, activation=leaky, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Dropout(0.2)(conv7)
    conv7 = Conv2D(128, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(64, 2, activation=leaky, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Dropout(0.2)(conv8)
    conv8 = Conv2D(64, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(32, 2, activation=leaky, padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(32, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Dropout(0.2)(conv9)
    conv9 = Conv2D(2, 3, activation=leaky, padding='same', kernel_initializer='he_normal')(conv9)

    out_dec = Conv2D(1, (1, 1), activation='sigmoid', padding='same', name="out_dec")(conv9)

    model = Model(inputs, outputs=[out_enc, out_dec])  

    return model 

  def create_generator(self):

    leaky = tf.keras.layers.LeakyReLU(self.alpha)

    # TODO: Try with generator equal to unet decoder
    inputs = Input(self.latent_size)

    dense1 = Dense(8*8*256, use_bias=False, input_shape=(self.latent_size,))(inputs)
    dense1 = BatchNormalization()(dense1)
    dense1 = leaky(dense1)
    dense1 = Reshape((8, 8, 256))(dense1)

    conv1 = Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(dense1)
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

    model = Model(inputs, output)
    return model

  class _unetGANModel(keras.Model):
    def __init__(self, discriminator, generator, latent_size):
        super(unetGAN._unetGANModel, self).__init__()
        self.discriminator = discriminator
        self.generator = generator
        self.latent_size = latent_size

        self.loss_tracker_generator = keras.metrics.Mean(name="gen_loss")
        self.loss_tracker_discriminator = keras.metrics.Mean(name="disc_loss")
    
    def call(self, x):
      return x

    def compile(self, discriminator_optimizer, generator_optimizer):
      super(unetGAN._unetGANModel, self).compile()
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
      if isinstance(data, tuple):
        images, _ = data
      else:
        images = data

      noise = tf.random.normal([images.shape[0], self.latent_size])
      with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = self.generator(noise, training=True)
        
        real_output_enc = self.discriminator(images, training=True)[0]
        fake_output_enc = self.discriminator(generated_images, training=True)[0]

        real_output_dec = self.discriminator(images, training=True)[1]
        fake_output_dec = self.discriminator(generated_images, training=True)[1]
        
        gen_loss = self.generator_loss(fake_output_enc) + self.generator_loss(fake_output_dec)
        
        disc_loss = self.discriminator_loss(real_output_enc, fake_output_enc) + self.discriminator_loss(real_output_dec, fake_output_dec)
      
      gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
      gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)
      
      self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
      self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

      # Compute metrics
      self.loss_tracker_generator.update_state(gen_loss)
      self.loss_tracker_discriminator.update_state(disc_loss)
      return {'gen_loss': self.loss_tracker_generator.result(), 'disc_loss': self.loss_tracker_discriminator.result()}
        
      def test_step(self, data):
        pass

      @property
      def metrics(self):
          # We list our `Metric` objects here so that `reset_states()` can be
          # called automatically at the start of each epoch
          # or at the start of `evaluate()`.
          # If you don't implement this property, you have to call
          # `reset_states()` yourself at the time of your choosing.
          return [self.loss_tracker_generator, self.loss_tracker_discriminator]

  def _build_model(self):
    self.generator = self.create_generator()
    self.discriminator = self.create_unet_discriminator()

    model = self._unetGANModel(generator=self.generator, discriminator=self.discriminator, latent_size=self.latent_size)

    self.generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5, clipvalue=5)
    self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

    model.compile(generator_optimizer=self.generator_optimizer, discriminator_optimizer=self.discriminator_optimizer)

    return model

  def train_model(self, train_ds, training_size, benchmark_noise):
    # set checkpoint directory
    checkpoint_prefix = os.path.join(self.checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer=self.model.generator_optimizer,
                                    discriminator_optimizer=self.model.discriminator_optimizer,
                                    model=self.model)

    # creating dictionaries for history and accuracy for the plots
    loss_history = {}
    loss_history['G_loss'] = []
    loss_history['D_loss'] = []

    print("Starting training of the Unet GAN model.")

    batchesPerEpoch = int(training_size / self.batch_size)
    print("Batches per epoch ", batchesPerEpoch)

    for epoch in range(self.n_epochs+1):
      # Keep track of the losses at each step
      epoch_gen_loss = []
      epoch_disc_loss = []

      print("Starting epoch ", epoch)

      for step in range(batchesPerEpoch+1):
        batch, _ = next(train_ds)
        gen_loss_step, disc_loss_step = self.model.train_on_batch(batch, batch)

        epoch_gen_loss.append(gen_loss_step)
        epoch_disc_loss.append(disc_loss_step)

        if step % self.logging_step == 0:
          print(f"\tLosses at step {step}:")
          print(f"\t\tGenerator Loss: {gen_loss_step}")
          print(f"\t\tDiscriminator Loss: {disc_loss_step}")

              
      if epoch % self.logging_step == 0:
        generator_images = self.model.generator(benchmark_noise, training=False)
                  
        print("Generated images: ")
        self.plot_fake_figures(generator_images, 4, epoch, self.out_images_path, "generated")

        print("Decoded maps: ")
        decoded_images = self.model.discriminator(generator_images, training=False)[1]
        self.plot_fake_figures(decoded_images, 4, epoch, self.out_images_path, "decoded")

        checkpoint.save(file_prefix = checkpoint_prefix)

  @staticmethod 
  def plot_fake_figures(x, n, epoch, img_dir,image_type):
      fig = plt.figure(figsize=(6,6))
      for i in range(n*n):
          plt.subplot(n,n,i+1)
          plt.xticks([])
          plt.yticks([])
          plt.grid(False)
          img=x[i,:,:,:]
          # rescale for visualization purposes
          img = tf.keras.preprocessing.image.array_to_img(img)
          plt.imshow(img, cmap="gray")
      plt.savefig('{}/{}_at_epoch_{:04d}.png'.format(img_dir, image_type, epoch))

      plt.show()

  def plot_losses(self, history):
      pd.DataFrame(history).plot(figsize=(10,8))
      plt.grid(True)
      plt.show()