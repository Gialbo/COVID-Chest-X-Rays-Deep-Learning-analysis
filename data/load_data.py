import tensorflow as tf

def load_data(dir, img_size, batch_size, generate_validation=True, generate_test=True, validation_split=0.2):

    # Load data from path and return the datasets (train, validation, test)

    #train_dir = '/content/drive/MyDrive/BIOINF/covid-project/COVID-19-xrays-resized/train'
    #test_dir = '/content/drive/MyDrive/BIOINF/covid-project/COVID-19-xrays-resized/test'

    img_height = img_size[0]
    img_width = img_size[1]

    train_dir = dir + '/train'
    test_dir = dir + '/test'

    if generate_validation == True:
        Train_Gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255, validation_split=validation_split)
        
        train_ds = Train_Gen.flow_from_directory(train_dir, 
                                                    target_size = (img_height, img_width), 
                                                    batch_size = batch_size, 
                                                    class_mode = 'binary',
                                                    subset='training')

        validation_ds = Train_Gen.flow_from_directory(train_dir, 
                                                        target_size = (img_height, img_width), 
                                                        batch_size = batch_size, 
                                                        class_mode = 'binary',
                                                        subset='validation')
    else: 
        Train_Gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
        train_ds = Train_Gen.flow_from_directory(train_dir, 
                                                    target_size = (img_height, img_width), 
                                                    batch_size = batch_size, 
                                                    class_mode = 'binary')

    Test_Gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale = 1./255)
    
    
    if generate_validation == True and generate_test == True:
        test_ds = Test_Gen.flow_from_directory(test_dir, 
                                                    target_size = (img_height, img_width),
                                                    class_mode = 'binary', 
                                                    batch_size = batch_size)
                                                    
        return train_ds, validation_ds, test_ds

    elif generate_validation == False and generate_test == False:
        return train_ds
