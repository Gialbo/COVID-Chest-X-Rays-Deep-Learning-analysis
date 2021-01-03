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

    Train_Gen = tf.keras.preprocessing.image.ImageDataGenerator(1./255)
    Test_Gen = tf.keras.preprocessing.image.ImageDataGenerator(1./255)

    train_ds = Train_Gen.flow_from_directory(trainFolder, 
                                            target_size = (imgHeight, imgWidth), 
                                            batch_size = trainBatchSize, 
                                            class_mode = 'input')



    # AUTOTUNE = tf.data.experimental.AUTOTUNE
    # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)

    test_ds = Train_Gen.flow_from_directory(testFolder, 
                                            target_size = (imgHeight, imgWidth), 
                                            batch_size = testBatchSize, 
                                            class_mode = 'input')

    model = Unet(pretrained_weights=None, input_size=input_shape)

    model.compile(optimizer = Adam(lr = lr), loss = 'binary_crossentropy')

    if verbose:
        model.summary()

    # Create a callback that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpointDir,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 save_freq=int(trainBatchSize/10))

    model.fit(train_ds, batch_size = trainBatchSize, epochs = epochs, callbacks=[cp_callback])

    generated_images = model.predict(test_ds)
    print(generated_images.shape)

__main__()