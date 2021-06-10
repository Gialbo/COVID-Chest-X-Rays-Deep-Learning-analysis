"""
Load x-rays dataset from directory 
Bioinformatics, Politecnico di Torino
Authors: Gilberto Manunza, Silvia Giammarinaro
"""


import tensorflow as tf
import os

class XRaysDataset():

    # LOAD TRAINING DATA

    def __init__(self,
                img_height=128,
                img_width=128,
                batch_size=128,
                train_dir='/content/drive/MyDrive/BIOINF/covid-project/dataset/train'):

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.train_dir = train_dir
    

    def preprocessing_function(self, x):
        x = tf.cast(x, tf.float32)
        return (x - 127.5)/127.5

    def process_path(self, file_path):
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = decode_img(img)
        return img

    def decode_img(self, image):
        # Convert the compressed string to a 3D uint8 tensor
        image = tf.image.decode_jpeg(image, channels=1)
        image = tf.image.resize(image, [self.img_height, self.img_width])
        return image

    def configure_for_performance(self, ds, buffer_size, batch_size):
        ds = ds.cache()
        ds = ds.shuffle(buffer_size=buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def get_file_paths(self, dir):
        file_paths = []
        for subdir, dirs, files in os.walk(dir):
            for file_name in files:
                file_path = os.path.join(subdir, file_name)
                file_paths.append(file_path)
        return file_paths

    def load(self):

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        
        training_size = len(os.listdir(self.train_dir+"/covid-19")) +len(os.listdir(self.train_dir+"/normal")) + len(os.listdir(self.train_dir+"/viral-pneumonia"))
        print("Training size ", training_size)

        covid_file_paths = get_file_paths(self.train_dir+"/covid-19")
        train_ds_covid = tf.data.Dataset.from_tensor_slices((covid_file_paths))
        train_ds_covid = train_ds_covid.map(process_path, num_parallel_calls=AUTOTUNE)
        train_ds_covid = train_ds_covid.map(preprocessing_function)
        train_ds_covid = configure_for_performance(train_ds_covid, buffer_size=1500, batch_size=self.batch_size)
        print(f"Number of batches for the covid dataset: {len(train_ds_covid)}")

        normal_file_paths = get_file_paths(self.train_dir+"/normal")
        train_ds_normal = tf.data.Dataset.from_tensor_slices((covid_file_paths))
        train_ds_normal = train_ds_normal.map(process_path, num_parallel_calls=AUTOTUNE)
        train_ds_normal = train_ds_normal.map(preprocessing_function)
        train_ds_normal = configure_for_performance(train_ds_normal, buffer_size=1500, batch_size=self.batch_size)
        print(f"Number of batches for the normal dataset: {len(train_ds_normal)}")

        vp_file_paths = get_file_paths(self.train_dir+"/viral-pneumonia")
        train_ds_vp = tf.data.Dataset.from_tensor_slices((covid_file_paths))
        train_ds_vp = train_ds_vp.map(process_path, num_parallel_calls=AUTOTUNE)
        train_ds_vp = train_ds_vp.map(preprocessing_function)
        train_ds_vp = configure_for_performance(train_ds_vp, buffer_size=1500, batch_size=self.batch_size)
        print(f"Number of batches for the viral pneumonia dataset: {len(train_ds_vp)}")

        train_datasets = [train_ds_covid, train_ds_normal, train_ds_vp]

        return train_datasets