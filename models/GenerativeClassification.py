"""
Generative Classification model
Bioinformatics, Politecnico di Torino
Authors: Gilberto Manunza, Silvia Giammarinaro
"""

import tensorflow as tf
import sys
from skimage.transform import resize
import numpy as np

class GenerativeClassification():
  """
  Classification task on data generated from a generative model.
  """ 
  def __init__(self,
              n_epochs=100,
              batch_size=1024,
              logging_step=10,
              generative_model=None,
              classification_model=None,
              half_data = False,
              checkpoint_dir = "/content/drive/MyDrive/BIOINF/checkpoints_classification/modelsDA_unc/",
              patience = 0):
    """
    It Accepts:
        - n_epochs
        - batch_size
        - logging_step: after how many epochs log information about generated images
        - generative_model_class: generative model to use
        - classification_model_class: classification model to use
        - half_data: if True use for training half batch of the training set and half batch of generated data, 
                     otherwise use one batch from the training set and one batch from the generated data.
        - checkpoint_dir
    """

    self.n_epochs = n_epochs
    self.batch_size = batch_size
    self.logging_step = logging_step
    self.generative_model = generative_model
    self.classification_model = classification_model
    self.half_data = half_data
    self.checkpoint_dir = checkpoint_dir
    self.patience = patience

    self.best_loss = sys.float_info.max

  # scale an array of images to a new size
  def scale_images(self, images, new_shape):
    images_list = list()
    for image in images:
      # resize with nearest neighbor interpolation
      new_image = resize(image, new_shape, 0)
      # store
      images_list.append(new_image)
    return np.asarray(images_list)

  def train_model(self, train_ds, val_ds, steps_per_epoch):
    history = {}
    history["train_loss"] = []
    history["train_acc"] = []
    history["train_recall_0"] = []
    history["train_recall_1"] = []
    history["train_recall_2"] = []
    history["train_precision_0"] = []
    history["train_precision_1"] = []
    history["train_precision_2"] = []
    history["val_loss"] = []
    history["val_acc"] = []
    history["val_recall_0"] = []
    history["val_recall_1"] = []
    history["val_recall_2"] = []
    history["val_precision_0"] = []
    history["val_precision_1"] = []
    history["val_precision_2"] = []
    step = 0
    wrs_epochs = 0
    for epoch in range(self.n_epochs):
      print(f"Starting epoch: {epoch+1}")

      for X_real, y_real in train_ds:
        Z, y_fake = self.generative_model.generate_latent_points()
        
        X_fake = self.generative_model.generator.predict([Z, y_fake])
        y_fake = tf.one_hot(y_fake, depth=3)

        X_fake_rgb = tf.image.grayscale_to_rgb(tf.convert_to_tensor(X_fake), name=None)
        X_fake_scaled = self.scale_images(X_fake_rgb, (224,224,3))

        X = tf.concat([X_real, X_fake_scaled], axis=0)
        y = tf.concat([y_real, y_fake], axis=0)
        
        indices = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
        shuffled_indices = tf.random.shuffle(indices)

        shuffled_X = tf.gather(X, shuffled_indices)
        shuffled_y = tf.gather(y, shuffled_indices)
        if self.half_data:
          shuffled_X = shuffled_X[int(shuffled_X.shape[0]/2):]
          shuffled_y = shuffled_y[int(shuffled_y.shape[0]/2):]

        results = self.classification_model.train_on_batch(shuffled_X, shuffled_y)
            
        history["train_loss"].append(results[0])
        history["train_acc"].append(results[1])
        history["train_recall_0"].append(results[2])
        history["train_recall_1"].append(results[3])
        history["train_recall_2"].append(results[4])
        history["train_precision_0"].append(results[5])
        history["train_precision_1"].append(results[6])
        history["train_precision_2"].append(results[7])
        step += 1

      print(f'\tTraining report at step {step} Loss: {history["train_loss"][-1]}, Accuracy: {history["train_acc"][-1]}')

      results = self.classification_model.evaluate(val_ds)
        
      history["val_loss"].append(results[0])
      history["val_acc"].append(results[1])
      history["val_recall_0"].append(results[2])
      history["val_recall_1"].append(results[3])
      history["val_recall_2"].append(results[4])
      history["val_precision_0"].append(results[5])
      history["val_precision_1"].append(results[6])
      history["val_precision_2"].append(results[7])
      
      wrs_epochs += 1
      # Early stopping based on best loss
      if results[0] < self.best_loss:
        self.best_loss = results[0]
        wrs_epochs = 0

        checkpoint_prefix = self.checkpoint_dir + "/best/ckpt"
        checkpoint = tf.train.Checkpoint(model=self.classification_model)
        checkpoint.save(file_prefix=checkpoint_prefix)

      print(f'\tValidation report at step {step} Loss: {history["val_loss"][-1]}, Accuracy: {history["val_acc"][-1]}')

      if wrs_epochs > self.patience:
        break


    checkpoint_prefix = self.checkpoint_dir + "/last/ckpt"
    checkpoint = tf.train.Checkpoint(model=self.classification_model)
    checkpoint.save(file_prefix=checkpoint_prefix)
    return history