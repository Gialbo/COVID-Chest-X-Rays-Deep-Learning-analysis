from keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import argparse
import sys

from unet.model.unet import Unet

def __main__():
    """
    Main entry point of the Unet, reads arguments, prepare datasets and run the algorihm 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDir', type=str, default='COVID-19-xrays/train', help='Folder of the train dataset')
    parser.add_argument('--testDir', type=str, default='COVID-19-xrays/test', help='Folder of the test dataset')
    parser.add_argument('--outDir', type=str, default='COVID-19-xrays/output', help='Output folder for storing generated images')
    parser.add_argument('--imageHeight', type=int,  default=128, help='Height of the image')
    parser.add_argument('--imageWidtht', type=int,  default=128, help='Width of the image')
    parser.add_argument('--useValidation', type=bool, default=True, help='Wheter or not to use validation. Note: Validation data will be split from the train')
    parser.add_argument('--valSplit', type=int, default=0.1, help='Amount of training data used for validation')
    parser.add_argument('--trainBatchSize', type=int, default=128, help='Batch size used for training')
    parser.add_argument('--valBatchSize', type=int, default=128, help='Batch size used for validation')
    parser.add_argument('--testBatchSize', type=int, default=128, help='Batch size used for testing')
    parser.add_argument('--epochsNumber', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--learningRate', type=float, default=1e-3, help='Learning rate')

    args = parser.parse_args()

    trainFolder = args.trainDir
    testFolder = args.testDir
    outFolder = args.outDir
    imgHeight = args.imageHeight
    imgWidth = args.imageWidth
    useVal = args.useValidation
    valSplit = args.valSplit
    trainBatchSize = args.trainBatchSize
    valBatchSize = args.valBatchSize
    testBatchSize = args.testBatchSize
    epochs = args.epochsNumber
    lr = args.learningRate

    input_shape = (imgHeight, imgWidth, 3)

    Train_Gen = tf.keras.preprocessing.image.ImageDataGenerator(1./255)
    Test_Gen = tf.keras.preprocessing.image.ImageDataGenerator(1./255)

    train_Generator = Train_Gen.flow_from_directory(trainFolder, 
                                                target_size = (imgHeight, imgWidth), 
                                                batch_size = trainBatchSize, 
                                                class_mode = 'binary')

    test_Generator = Test_Gen.flow_from_directory(testFolder, 
                                              target_size = (imgHeight, imgWidth),
                                              class_mode = 'binary', 
                                              batch_size = testBatchSize)

    model = Unet(pretrained_weights=None, input_size=input_shape)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
