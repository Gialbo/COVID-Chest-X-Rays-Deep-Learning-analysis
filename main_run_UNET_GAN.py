from keras.optimizers import Adam
import tensorflow as tf
import argparse
import matplotlib.pyplot as plt

from models.unet import Unet
from models.generator import Generator

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, generator, discriminator, trainBatchSize, noise_dim, generator_optimizer, discriminator_optimizer):
    noise = tf.random.normal([trainBatchSize, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output_enc = discriminator(images, training=True)[0]
        fake_output_enc = discriminator(generated_images, training=True)[0]

        
        real_output_dec = discriminator(images, training=True)[1]
        fake_output_dec = discriminator(generated_images, training=True)[1]

        gen_loss = generator_loss(fake_output_enc) + generator_loss(fake_output_dec)

        disc_loss = discriminator_loss(real_output_enc, fake_output_enc) + discriminator_loss(real_output_dec, fake_output_dec)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def __main__():
    """
    Main entry point of the Unet-GAN, reads arguments, prepare datasets and run the algorihm 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDir', type=str, default='./COVID-19-xrays-sample/train', help='Folder of the train dataset')
    parser.add_argument('--testDir', type=str, default='./COVID-19-xrays-sample/test', help='Folder of the test dataset')
    parser.add_argument('--outDir', type=str, default='./COVID-19-xrays-sample/output', help='Output folder for storing generated images')
    parser.add_argument('--checkpointDir', type=str, default='/checkpoints', help='Folder for saving the checkpoints')
    parser.add_argument('--imageHeight', type=int,  default=128, help='Height of the image')
    parser.add_argument('--imageWidth', type=int,  default=128, help='Width of the image')
    parser.add_argument('--trainBatchSize', type=int, default=128, help='Batch size used for training')
    parser.add_argument('--testBatchSize', type=int, default=128, help='Batch size used for testing')
    parser.add_argument('--epochsNumber', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--learningRate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--verbose', type=bool, default=True, help='True to log some informations about the model')

    args = parser.parse_args()

    trainFolder = args.trainDir
    testFolder = args.testDir
    outFolder = args.outDir
    checkpointDir =args.checkpointDir
    imgHeight = args.imageHeight
    imgWidth = args.imageWidth
    trainBatchSize = args.trainBatchSize
    testBatchSize = args.testBatchSize
    epochs = args.epochsNumber
    lr = args.learningRate
    verbose = args.verbose

    input_shape = (imgHeight, imgWidth, 3)


    Train_Gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
    Test_Gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)

    train_ds = Train_Gen.flow_from_directory(trainFolder, 
                                            target_size = (imgHeight, imgWidth), 
                                            batch_size = trainBatchSize, 
                                            class_mode = 'input')

    test_ds = Test_Gen.flow_from_directory(testFolder, 
                                            target_size = (imgHeight, imgWidth), 
                                            batch_size = testBatchSize, 
                                            class_mode = 'input')

    noise_dim = 100

    discriminator = Unet(pretrained_weights=None, input_size=input_shape)

    generator = Generator()

    generator_optimizer = tf.keras.optimizers.Adam(1e-4)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    # Create a callback that saves the model's weights

    for epoch in range(epochs):
        print(f"Starting epoch {epoch}")
        for batch in train_ds.next():
            train_step(batch, generator, discriminator, trainBatchSize, noise_dim, generator_optimizer, discriminator_optimizer)
            generated_image = generator.predict(tf.random.normal([1, noise_dim]))

            if epoch % 10 == 0:
                img = tf.keras.preprocessing.image.array_to_img(generated_image[0])
                img.save(outFolder+f"/generated_img_epoch_{epoch}.png")

                img = tf.keras.preprocessing.image.array_to_img(discriminator(generated_image[0], training=False)[1])
                img.save(outFolder+f"/decoded_img_epoch_{epoch}.png")


    generated_images = generator.predict(tf.random.normal([1, noise_dim]))

    for i, image in enumerate(generated_images):
      img = tf.keras.preprocessing.image.array_to_img(image)
      img.save(outFolder+f"/generate_img_{i}.png")

__main__()