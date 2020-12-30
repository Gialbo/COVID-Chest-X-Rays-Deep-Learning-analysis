from keras.optimizers import Adam
import tensorflow as tf
import matplotlib.pyplot as plt
import random
import argparse

from models.unet import Unet

def __main__():
    """
    Main entry point of the Unet, reads arguments, prepare datasets and run the algorihm 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDir', type=str, default='./COVID-19-xrays-sample/train', help='Folder of the train dataset')
    parser.add_argument('--testDir', type=str, default='./COVID-19-xrays-sample/test', help='Folder of the test dataset')
    parser.add_argument('--outDir', type=str, default='./COVID-19-xrays-sample/output', help='Output folder for storing generated images')
    parser.add_argument('--checkpointDir', type=str, default='/checkpoints', help='Folder for saving the checkpoints')
    parser.add_argument('--imageHeight', type=int,  default=128, help='Height of the image')
    parser.add_argument('--imageWidth', type=int,  default=128, help='Width of the image')
    parser.add_argument('--useValidation', type=bool, default=True, help='Wheter or not to use validation. Note: Validation data will be split from the train')
    parser.add_argument('--valSplit', type=int, default=0.1, help='Amount of training data used for validation')
    parser.add_argument('--trainBatchSize', type=int, default=128, help='Batch size used for training')
    parser.add_argument('--valBatchSize', type=int, default=128, help='Batch size used for validation')
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
    useVal = args.useValidation
    valSplit = args.valSplit
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    testBatchSize = args.testBatchSize
    epochs = args.epochsNumber
    lr = args.learningRate
    verbose = args.verbose

    input_shape = (imgHeight, imgWidth, 3)

    Train_Gen = tf.keras.preprocessing.image.ImageDataGenerator(1./255)
    Test_Gen = tf.keras.preprocessing.image.ImageDataGenerator(1./255)

    train_ds = tf.data.Dataset.from_generator(
        lambda: Train_Gen.flow_from_directory(trainFolder, 
                                                target_size = (imgHeight, imgWidth), 
                                                batch_size = trainBatchSize, 
                                                class_mode = 'input',
                                                subset='training'), 
        (tf.float32, tf.int32))



    AUTOTUNE = tf.data.experimental.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    validation_ds = tf.data.Dataset.from_generator(
        lambda: Train_Gen.flow_from_directory(trainFolder, 
                                                target_size = (imgHeight, imgWidth), 
                                                batch_size = valBatchSize, 
                                                class_mode = 'input',
                                                subset='validation'), 
        (tf.float32, tf.int32))



    validation_ds = validation_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = tf.data.Dataset.from_generator(
        lambda: Test_Gen.flow_from_directory(trainFolder, 
                                                target_size = (imgHeight, imgWidth), 
                                                batch_size = testBatchSize, 
                                                class_mode = 'input'), 
        (tf.float32, tf.int32))



    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    model = Unet(pretrained_weights=None, input_size=input_shape)

    model.compile(optimizer = Adam(lr = lr), loss = 'binary_crossentropy')

    if verbose:
        model.summary()

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointDir,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=int(trainBatchSize/10))

    # model.fit(train_ds, batch_size = trainBatchSize, epochs = epochs, callbacks=[cp_callback])

    generated_images = model.predict(test_ds)
    print(test_ds.shape)

    # n = 10  # How many digits we will display
    # plt.figure(figsize=(20, 4))
    # for batch in generated_images.next():
    #     i = random.randint(0, valBatchSize)
    #     # Display original
    #     ax = plt.subplot(2, n, i + 1)
    #     plt.imshow(x_test[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)   
    #     # Display reconstruction
    #     ax = plt.subplot(2, n, i + 1 + n)
    #     plt.imshow(decoded_imgs[i].reshape(28, 28))
    #     plt.gray()
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)
    # plt.show()


__main__()