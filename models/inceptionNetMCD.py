import tensorflow as tf
import keras
from inceptionV3MCD import InceptionV3MCD
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Recall, Precision

def inceptionNetMCD(rate=0.3):

  labels = ["covid-19", "normal", "viral-pneumonia"]

  basemodel = InceptionV3MCD(
    rate = rate,
    include_top = False, 
    weights = 'imagenet', 
    input_tensor = Input((224, 224, 3))
  )

  basemodel.trainable = True

  # Add classification head to the model
  headmodel = basemodel.output
  headmodel = GlobalAveragePooling2D()(headmodel)
  headmodel = Flatten()(headmodel) 
  headmodel = Dense(256, activation = "relu")(headmodel)
  headmodel = Dropout(rate)(headmodel, training=True)
  headmodel = Dense(128, activation = "relu")(headmodel)
  headmodel = Dropout(rate)(headmodel, training=True)
  headmodel = Dense(3, activation = "softmax")(headmodel) # 3 classes

  model = Model(inputs = basemodel.input, outputs = headmodel)

  list_metrics = ["accuracy"]
  list_metrics += [Recall(class_id = i) for i in range(len(labels))] 
  list_metrics += [Precision(class_id = i) for i in range(len(labels))]

  model.compile(
      loss = "categorical_crossentropy",
      optimizer = "adam",
      metrics = list_metrics
  )

  return model