"""
Load chest x-rays dataset from directory 
Bioinformatics, Politecnico di Torino
Authors: Gilberto Manunza, Silvia Giammarinaro
"""


import tensorflow as tf
import os

class XRaysDataset():

    def __init__(self,
                img_height=128,
                img_width=128,
                batch_size=128,
                dir='/content/drive/MyDrive/BIOINF/covid-project/dataset/train'):

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.dir = dir
    

    def preprocessing_function(self, x):
        x = tf.cast(x, tf.float32)
        return (x - 127.5)/127.5

    def process_path(self, file_path):
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
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

    def load(self, train=True):
        # train = False is used to compute FID

        AUTOTUNE = tf.data.experimental.AUTOTUNE

        size = len(os.listdir(self.dir+"/covid-19")) +len(os.listdir(self.dir+"/normal")) + len(os.listdir(self.dir+"/viral-pneumonia"))
        print("Dataset size ", size)

        if train:
            
            covid_file_paths = self.get_file_paths(self.dir+"/covid-19")
            train_ds_covid = tf.data.Dataset.from_tensor_slices((covid_file_paths))
            train_ds_covid = train_ds_covid.map(self.process_path, num_parallel_calls=AUTOTUNE)
            train_ds_covid = train_ds_covid.map(self.preprocessing_function)
            train_ds_covid = self.configure_for_performance(train_ds_covid, buffer_size=1500, batch_size=self.batch_size)
            print(f"Number of batches for the covid dataset: {len(train_ds_covid)}")

            normal_file_paths = self.get_file_paths(self.dir+"/normal")
            train_ds_normal = tf.data.Dataset.from_tensor_slices((covid_file_paths))
            train_ds_normal = train_ds_normal.map(self.process_path, num_parallel_calls=AUTOTUNE)
            train_ds_normal = train_ds_normal.map(self.preprocessing_function)
            train_ds_normal = self.configure_for_performance(train_ds_normal, buffer_size=1500, batch_size=self.batch_size)
            print(f"Number of batches for the normal dataset: {len(train_ds_normal)}")

            vp_file_paths = self.get_file_paths(self.dir+"/viral-pneumonia")
            train_ds_vp = tf.data.Dataset.from_tensor_slices((covid_file_paths))
            train_ds_vp = train_ds_vp.map(self.process_path, num_parallel_calls=AUTOTUNE)
            train_ds_vp = train_ds_vp.map(self.preprocessing_function)
            train_ds_vp = self.configure_for_performance(train_ds_vp, buffer_size=1500, batch_size=self.batch_size)
            print(f"Number of batches for the viral pneumonia dataset: {len(train_ds_vp)}")

            ds = [train_ds_covid, train_ds_normal, train_ds_vp]

        else:

            label_mapping = {"covid-19": 0, "normal": 1, "viral-pneumonia": 2}
            file_paths, labels = self.get_file_paths(self.dir, label_mapping)
            test_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
            test_ds = test_ds.map(self.process_path)
            test_ds = test_ds.map(self.preprocessing_function)
            ds = configure_for_performance(test_ds, buffer_size=1500, batch_size=self.batch_size)
            print(f"Number of batches for the dataset: {len(ds)}")

        return ds, size