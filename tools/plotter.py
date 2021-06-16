"""
Utility functions for plotting results
Bioinformatics, Politecnico di Torino
Authors: Gilberto Manunza, Silvia Giammarinaro
"""

import matplotlib.pyplot as plt

def plot_losses(history):
  train_loss = history['3'].history["loss"]
  valid_loss = history['3'].history["val_loss"]

  epochs = range(len(train_loss)) 
  plt.figure(figsize=(8,6))
  with plt.style.context('fivethirtyeight'):
    plt.plot(epochs, train_loss)
    plt.plot(epochs, valid_loss)
    plt.legend(["Training Loss", "Validation Loss"])
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def plot_accuracies(history):
  train_acc = history['3'].history["accuracy"]
  valid_acc = history['3'].history["val_accuracy"]

  epochs = range(len(train_acc)) 
  plt.figure(figsize=(8,6))

  with plt.style.context('fivethirtyeight'):
    plt.plot(epochs, train_acc)
    plt.plot(epochs, valid_acc)
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

def plot_recalls(history):
  with plt.style.context('fivethirtyeight'):
    fig, axs = plt.subplots(1, 3, figsize = (15, 5))

    train_rec_0 = history['3'].history["recall_6"]
    valid_rec_0 = history['3'].history["val_recall_6"]
    train_rec_1 = history['3'].history["recall_7"]
    valid_rec_1 = history['3'].history["val_recall_7"]
    train_rec_2 = history['3'].history["recall_8"]
    valid_rec_2 = history['3'].history["val_recall_8"]

    epochs = range(len(train_rec_0)) 


    axs[0].plot(epochs, train_rec_0)
    axs[0].plot(epochs, valid_rec_0)
    axs[0].legend(["Training Recall", "Validation Recall"])
    axs[0].set_title("Recall, Covid-19")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Recall")

    axs[1].plot(epochs, train_rec_1)
    axs[1].plot(epochs, valid_rec_1)
    axs[1].legend(["Training Recall", "Validation Recall"])
    axs[1].set_title("Recall, Normal")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Recall")

    axs[2].plot(epochs, train_rec_2)
    axs[2].plot(epochs, valid_rec_2)
    axs[2].legend(["Training Recall", "Validation Recall"])
    axs[2].set_title("Recall, Viral Pneumonia")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Recall")

    fig.tight_layout()

def plot_precisions(history):
  with plt.style.context('fivethirtyeight'):
    fig, axs = plt.subplots(1, 3, figsize = (15, 5))

    train_pre_0 = history['3'].history["precision_6"]
    valid_pre_0 = history['3'].history["val_precision_6"]
    train_pre_1 = history['3'].history["precision_7"]
    valid_pre_1 = history['3'].history["val_precision_7"]
    train_pre_2 = history['3'].history["precision_8"]
    valid_pre_2 = history['3'].history["val_precision_8"]

    epochs = range(len(train_pre_0)) 

    axs[0].plot(epochs, train_pre_0)
    axs[0].plot(epochs, valid_pre_0)
    axs[0].legend(["Training Precision", "Validation Precision"])
    axs[0].set_title("Precision, Covid-19")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Precision")

    axs[1].plot(epochs, train_pre_1)
    axs[1].plot(epochs, valid_pre_1)
    axs[1].legend(["Training Precision", "Validation Precision"])
    axs[1].set_title("Precision, Normal")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Precision")

    axs[2].plot(epochs, train_pre_2)
    axs[2].plot(epochs, valid_pre_2)
    axs[2].legend(["Training Precision", "Validation Precision"])
    axs[2].set_title("Precision, Viral Pneumonia")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Precision")

    fig.tight_layout()

def plot_classification_report(history):
  plot_losses(history)
  plot_accuracies(history)
  plot_recalls(history)
  plot_precisions(history)