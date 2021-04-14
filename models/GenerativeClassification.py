import tensorflow as tf
import numpy as np

class GenerativeClassification():
    """
    Permits to perform a classification task on data generated from a generative model. It is possible to
    specify how much of the dataset should be generated and how much should be loaded from the real one.
    """
  
    def __init__(self,
                n_epochs=100,
                batch_size=256,
                logging_step=10,
                generative_model=None,
                classification_model=None,
                checkpoint_dir="checkpoints"):
        """
        It Accepts:
            - n_epochs
            - batch_size
            - logging_step: after how many epochs log information about generated images
            - generative_model_class: generative model to use
            - classification_model_class: classification model to use
            - real_data_train_prop: proportion of real data to use during training. Accepts values in range [0, 1], with 0 meaning use
                                    generated data and 1 meaning use only real data
            - real_data_val_propr: same as above but for the validation set
            - real_Data_test_prop: same as above but for the test set
            - checkpoint_dir
        """
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.logging_step = logging_step
        self.generative_model = generative_model
        self.classification_model = classification_model
        self.checkpoints_dir = checkpoint_dir

    def train_model(self, val_ds, steps_per_epoch):

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

        for epoch in range(self.n_epochs):
            print(f"Starting epoch: {epoch}")
            for _ in range(steps_per_epoch):
                Z, y = self.generative_model.generate_latent_points()
                X = self.generative_model.generator.predict([Z, y])
                y = tf.one_hot(y, depth=3)

                results = self.classification_model.train_on_batch(X, y)
                
                history["train_loss"].append(results[0])
                history["train_acc"].append(results[1])
                history["train_recall_0"].append(results[2])
                history["train_recall_1"].append(results[3])
                history["train_recall_2"].append(results[4])
                history["train_precision_0"].append(results[5])
                history["train_precision_1"].append(results[6])
                history["train_precision_2"].append(results[7])

            print(f'\tTraining report at epoch {epoch} Loss: {np.array(history["train_loss"]).mean()}, Accuracy: {np.array(history["train_acc"]).mean()}')

            results = self.classification_model.evaluate(val_ds)
            
            history["val_loss"].append(results[0])
            history["val_acc"].append(results[1])
            history["val_recall_0"].append(results[2])
            history["val_recall_1"].append(results[3])
            history["val_recall_2"].append(results[4])
            history["val_precision_0"].append(results[5])
            history["val_precision_1"].append(results[6])
            history["val_precision_2"].append(results[7])

            print(f'\tValidation report at epoch {epoch} Loss: {np.array(history["val_loss"]).mean()}, Accuracy: {np.array(history["val_acc"]).mean()}')