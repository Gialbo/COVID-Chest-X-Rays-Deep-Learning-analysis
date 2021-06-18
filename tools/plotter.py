"""
Utility functions for plotting results
Bioinformatics, Politecnico di Torino
Authors: Gilberto Manunza, Silvia Giammarinaro
"""

import matplotlib.pyplot as plt
import json

def plot_losses(history, isMCD=False, selected_run='3'):
  if isMCD:
    train_loss = history.history["loss" if "loss" in history.history.keys() else "train_loss"]
    valid_loss = history.history["val_loss"]
  else:
    train_loss = history[selected_run].history["loss" if "loss" in history[selected_run].history.keys() else "train_loss"]
    valid_loss = history[selected_run].history["val_loss"]

  if len(valid_loss) != len(train_loss):
    ind = int(len(train_loss)/len(valid_loss))-1
    train_loss = train_loss[ind::ind]
  epochs = range(len(valid_loss)) 
  plt.figure(figsize=(8,6))
  with plt.style.context('fivethirtyeight'):
    plt.plot(epochs, train_loss[:len(valid_loss)])
    plt.plot(epochs, valid_loss)
    plt.legend(["Training Loss", "Validation Loss"])
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

def plot_accuracies(history, isMCD=False, selected_run='3'):
  if isMCD:
    train_acc = history.history["accuracy"]
    valid_acc = history.history["val_accuracy"]
  else:
    train_acc = history[selected_run].history["accuracy" if "accuracy" in history[selected_run].history.keys() else "train_acc"]
    valid_acc = history[selected_run].history["val_accuracy" if "val_accuracy" in history[selected_run].history.keys() else "val_acc"]

  if len(valid_acc) != len(train_acc):
    ind = int(len(train_acc)/len(valid_acc))-1
    train_acc = train_acc[ind::ind]
  epochs = range(len(valid_acc)) 
  plt.figure(figsize=(8,6))

  with plt.style.context('fivethirtyeight'):
    plt.plot(epochs, train_acc[:len(valid_acc)])
    plt.plot(epochs, valid_acc)
    plt.legend(["Training Accuracy", "Validation Accuracy"])
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.show()

def plot_recalls(history, isMCD=False, selected_run='3'):
  with plt.style.context('fivethirtyeight'):
    fig, axs = plt.subplots(1, 3, figsize = (15, 5))

    if isMCD:
      keys = list(history.history.keys())
      train_rec_0 = history.history[keys[2]]
      valid_rec_0 = history.history[keys[10]]
      train_rec_1 = history.history[keys[3]]
      valid_rec_1 = history.history[keys[11]]
      train_rec_2 = history.history[keys[4]]
      valid_rec_2 = history.history[keys[12]]
    else:
      keys = list(history[selected_run].history.keys())
      train_rec_0 = history[selected_run].history[keys[2]]
      valid_rec_0 = history[selected_run].history[keys[10]]
      train_rec_1 = history[selected_run].history[keys[3]]
      valid_rec_1 = history[selected_run].history[keys[11]]
      train_rec_2 = history[selected_run].history[keys[4]]
      valid_rec_2 = history[selected_run].history[keys[12]]

    if len(valid_rec_0) != len(train_rec_0):
      ind_0 = int(len(train_rec_0)/len(valid_rec_0))-1
      ind_1 = int(len(train_rec_1)/len(valid_rec_1))-1
      ind_2 = int(len(train_rec_2)/len(valid_rec_2))-1
      train_rec_0 = train_rec_0[ind_0::ind_0]
      train_rec_1 = train_rec_1[ind_1::ind_1]
      train_rec_2 = train_rec_2[ind_2::ind_2]

    epochs = range(len(valid_rec_0)) 


    axs[0].plot(epochs, train_rec_0[:len(valid_rec_0)])
    axs[0].plot(epochs, valid_rec_0)
    axs[0].legend(["Training Recall", "Validation Recall"])
    axs[0].set_title("Recall, Covid-19")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Recall")

    axs[1].plot(epochs, train_rec_1[:len(valid_rec_0)])
    axs[1].plot(epochs, valid_rec_1)
    axs[1].legend(["Training Recall", "Validation Recall"])
    axs[1].set_title("Recall, Normal")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Recall")

    axs[2].plot(epochs, train_rec_2[:len(valid_rec_0)])
    axs[2].plot(epochs, valid_rec_2)
    axs[2].legend(["Training Recall", "Validation Recall"])
    axs[2].set_title("Recall, Viral Pneumonia")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Recall")

    fig.tight_layout()

def plot_precisions(history, isMCD=False, selected_run='3'):
  with plt.style.context('fivethirtyeight'):
    fig, axs = plt.subplots(1, 3, figsize = (15, 5))

    if isMCD:
      keys = list(history.history.keys())
      train_pre_0 = history.history[keys[5]]
      valid_pre_0 = history.history[keys[13]]
      train_pre_1 = history.history[keys[6]]
      valid_pre_1 = history.history[keys[14]]
      train_pre_2 = history.history[keys[7]]
      valid_pre_2 = history.history[keys[15]]
    else:
      keys = list(history[selected_run].history.keys())
      train_pre_0 = history[selected_run].history[keys[5]]
      valid_pre_0 = history[selected_run].history[keys[13]]
      train_pre_1 = history[selected_run].history[keys[6]]
      valid_pre_1 = history[selected_run].history[keys[14]]
      train_pre_2 = history[selected_run].history[keys[7]]
      valid_pre_2 = history[selected_run].history[keys[15]]

    if len(valid_pre_0) != len(train_pre_0):
      ind_0 = int(len(train_pre_0)/len(valid_pre_0))-1
      ind_1 = int(len(train_pre_1)/len(valid_pre_1))-1
      ind_2 = int(len(train_pre_2)/len(valid_pre_2))-1
      train_pre_0 = train_pre_0[ind_0::ind_0]
      train_pre_1 = train_pre_1[ind_1::ind_1]
      train_pre_2 = train_pre_2[ind_2::ind_2]
    epochs = range(len(valid_pre_0)) 

    axs[0].plot(epochs, train_pre_0[:len(valid_pre_0)])
    axs[0].plot(epochs, valid_pre_0)
    axs[0].legend(["Training Precision", "Validation Precision"])
    axs[0].set_title("Precision, Covid-19")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Precision")

    axs[1].plot(epochs, train_pre_1[:len(valid_pre_0)])
    axs[1].plot(epochs, valid_pre_1)
    axs[1].legend(["Training Precision", "Validation Precision"])
    axs[1].set_title("Precision, Normal")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Precision")

    axs[2].plot(epochs, train_pre_2[:len(valid_pre_0)])
    axs[2].plot(epochs, valid_pre_2)
    axs[2].legend(["Training Precision", "Validation Precision"])
    axs[2].set_title("Precision, Viral Pneumonia")
    axs[2].set_xlabel("Epochs")
    axs[2].set_ylabel("Precision")

    fig.tight_layout()

def plot_classification_report(history, isMCD=False, selected_run='3'):
  plot_losses(history, isMCD, selected_run=selected_run)
  plot_accuracies(history, isMCD, selected_run=selected_run)
  plot_recalls(history, isMCD, selected_run=selected_run)
  plot_precisions(history, isMCD, selected_run=selected_run)

def save_history(history, checkpoint_dir):
  with open(f"{checkpoint_dir}.json", 'w') as fs:
    json.dump(history, fs)
