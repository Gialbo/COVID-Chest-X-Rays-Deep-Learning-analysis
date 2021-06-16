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
                dir='/content/drive/MyDrive/BIOINF/covid-project/dataset/train',
                isInceptionNet=False):

        self.img_height = img_height
        self.img_width = img_width
        self.batch_size = batch_size
        self.dir = dir
        self.isInceptionNet = isInceptionNet
    

    def preprocessing_function(self, x, label=None):
        x = tf.cast(x, tf.float32)
        if label is not None:
            return (x - 127.5)/127.5, label
        else:
            return (x - 127.5)/127.5

    def process_path(self, file_path, label=None):
        # Load the raw data from the file as a string
        img = tf.io.read_file(file_path)
        img = self.decode_img(img)
        if label is not None:
            if self.isInceptionNet:
                label = tf.one_hot(label, depth=3)
            
            return img, label
        else:
            return img

    def decode_img(self, image):
        # Convert the compressed string to a 3D uint8 tensor
        image = tf.image.decode_jpeg(image, channels=1)
        if self.isInceptionNet:
            image = tf.image.grayscale_to_rgb(image, name=None)
        image = tf.image.resize(image, [self.img_height, self.img_width])
        return image

    def configure_for_performance(self, ds, buffer_size, batch_size, shuffle=True):
        ds = ds.cache()
        if shuffle:
            ds = ds.shuffle(buffer_size=buffer_size)
        ds = ds.batch(self.batch_size)
        ds = ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return ds

    def configure_for_performance_train_val(self, ds, buffer_size, batch_size):
        VAL_SPLIT=0.2
        ds = ds.shuffle(buffer_size=buffer_size)
        val_ds = ds.take(int(len(ds)*VAL_SPLIT))
        train_ds = ds.skip(int(len(ds)*VAL_SPLIT))

        val_ds = val_ds.cache()
        val_ds = val_ds.batch(batch_size)
        val_ds = val_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

        train_ds = train_ds.cache()
        train_ds = train_ds.batch(batch_size)
        train_ds = train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return train_ds, val_ds


    def get_file_paths(self, dir, label_mapping=None):
        file_paths = []
        if label_mapping:
            labels = []
            for subdir, dirs, files in os.walk(dir):
                for file_name in files:
                    file_path = os.path.join(subdir, file_name)
                    file_paths.append(file_path)
                    labels.append(label_mapping[subdir.split("/")[-1]])
            return file_paths, labels
        else:
            for subdir, dirs, files in os.walk(dir):
                for file_name in files:
                    file_path = os.path.join(subdir, file_name)
                    file_paths.append(file_path)
            return file_paths

    def load(self, separate_classes=True, train_val_split=False, covid_class=False, shuffle=True):
        # separate_classes = False is used to load the entire training/test dataset

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        label_mapping = {"covid-19": 0, "normal": 1, "viral-pneumonia": 2}

        if shuffle:
            file_paths, labels = self.get_file_paths(self.dir, label_mapping)
            ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
            ds = ds.map(self.process_path)
            ds = ds.map(self.preprocessing_function)
            ds = self.configure_for_performance(ds, buffer_size=1500, batch_size=self.batch_size, shuffle=shuffle)
            print(f"Number of batches for the dataset: {len(ds)}")


        if train_val_split:
            
            file_paths, labels = self.get_file_paths(self.dir, label_mapping)
            train_val_ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
            train_val_ds = train_val_ds.map(self.process_path)
            train_val_ds = train_val_ds.map(self.preprocessing_function)
            train_ds, val_ds = self.configure_for_performance_train_val(train_val_ds, buffer_size=3443, batch_size=self.batch_size)
            print(f"Number of batches for the train dataset: {len(train_ds)}")
            print(f"Number of batches for the validation dataset: {len(val_ds)}")
            return train_ds, val_ds
        
        if covid_class:

            covid_file_paths = self.get_file_paths(self.dir+"/covid-19")
            train_ds_covid = tf.data.Dataset.from_tensor_slices((covid_file_paths))
            train_ds_covid = train_ds_covid.map(self.process_path, num_parallel_calls=AUTOTUNE)
            train_ds_covid = train_ds_covid.map(self.preprocessing_function)
            train_ds_covid = self.configure_for_performance(train_ds_covid, buffer_size=1500, batch_size=self.batch_size)
            print(f"Number of batches for the covid dataset: {len(train_ds_covid)}")
            size_covid = len(os.listdir(self.dir+"/covid-19"))
            return train_ds_covid, size_covid

        size = len(os.listdir(self.dir+"/covid-19")) +len(os.listdir(self.dir+"/normal")) + len(os.listdir(self.dir+"/viral-pneumonia"))
        print("Dataset size ", size)

        if separate_classes:
            
            covid_file_paths = self.get_file_paths(self.dir+"/covid-19")
            train_ds_covid = tf.data.Dataset.from_tensor_slices((covid_file_paths))
            train_ds_covid = train_ds_covid.map(self.process_path, num_parallel_calls=AUTOTUNE)
            train_ds_covid = train_ds_covid.map(self.preprocessing_function)
            train_ds_covid = self.configure_for_performance(train_ds_covid, buffer_size=1500, batch_size=self.batch_size)
            print(f"Number of batches for the covid dataset: {len(train_ds_covid)}")

            normal_file_paths = self.get_file_paths(self.dir+"/normal")
            train_ds_normal = tf.data.Dataset.from_tensor_slices((normal_file_paths))
            train_ds_normal = train_ds_normal.map(self.process_path, num_parallel_calls=AUTOTUNE)
            train_ds_normal = train_ds_normal.map(self.preprocessing_function)
            train_ds_normal = self.configure_for_performance(train_ds_normal, buffer_size=1500, batch_size=self.batch_size)
            print(f"Number of batches for the normal dataset: {len(train_ds_normal)}")

            vp_file_paths = self.get_file_paths(self.dir+"/viral-pneumonia")
            train_ds_vp = tf.data.Dataset.from_tensor_slices((vp_file_paths))
            train_ds_vp = train_ds_vp.map(self.process_path, num_parallel_calls=AUTOTUNE)
            train_ds_vp = train_ds_vp.map(self.preprocessing_function)
            train_ds_vp = self.configure_for_performance(train_ds_vp, buffer_size=1500, batch_size=self.batch_size)
            print(f"Number of batches for the viral pneumonia dataset: {len(train_ds_vp)}")

            ds = [train_ds_covid, train_ds_normal, train_ds_vp]

        else:
            file_paths, labels = self.get_file_paths(self.dir, label_mapping)
            ds = tf.data.Dataset.from_tensor_slices((file_paths, labels))
            ds = ds.map(self.process_path)
            ds = ds.map(self.preprocessing_function)
            ds = self.configure_for_performance(ds, buffer_size=1500, batch_size=self.batch_size)
            print(f"Number of batches for the dataset: {len(ds)}")

        return ds, size