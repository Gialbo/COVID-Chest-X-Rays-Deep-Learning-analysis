"""
Utility functions to compute FID, Frechet Inception Distance
Bioinformatics, Politecnico di Torino
Authors: Gilberto Manunza, Silvia Giammarinaro
"""

import numpy as np
from numpy import cov, trace, iscomplexobj, asarray
from numpy.random import randint
from scipy.linalg import sqrtm
from skimage.transform import resize
import tensorflow as tf


class FID():

    def scale_images(images, new_shape):
        images_list = list()
        for image in images:
            # resize with nearest neighbor interpolation
            new_image = resize(image, new_shape, 0)
            # store
            images_list.append(new_image)
        return asarray(images_list)

    # calculate frechet inception distance
    def calculate_fid(model, images1, images2):
        # calculate activations
        act1 = model.predict(images1)
        act2 = model.predict(images2)
        # calculate mean and covariance statistics
        mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
        mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
        # calculate sum squared difference between means
        ssdiff = numpy.sum((mu1 - mu2)**2.0)
        # calculate sqrt of product between cov
        covmean = sqrtm(sigma1.dot(sigma2))
        # check and correct imaginary numbers from sqrt
        if iscomplexobj(covmean):
            covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
        return fid

    def get_FID(batch_size, generative_model, evaluation_model, real_images, n_classes=3, n_trial=5):

        FID_array = np.zeros(trials)
        for i in n_trails:
            benchmarkNoise = tf.random.normal([batch_size, generative_model.latent_size])
            benchmarkLabels = np.random.randint(0, n_classes, batch_size)

            fake_images = generative_model.generator.predict([benchmarkNoise, benchmarkLabels])

            real_scaled_images = self.scale_images(real_images, (299,299,3))
            fake_images_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(fake_images), name=None)
            fake_scaled_images = self.scale_images(fake_images_rgb, (299,299,3))


            fid = self.calculate_fid(evaluation_model, real_scaled_images, fake_scaled_images)
            FID_array[i] = fid

        mean_FID = np.mean(FID_array)
        std_FID = np.std(FID_array)

        return mean_FID, std_FID

                
 