"""
Utility functions to compute uncertainty for deterministic and Monte Carlo Dropout models
Bioinformatics, Politecnico di Torino
Authors: Gilberto Manunza, Silvia Giammarinaro
"""

import matplotlib.pyplot as plt
import seaborn as sns

def compute_uncertainties(net, X, MC=100):
    # MC times predictions for a single image
    p_hat = list()
    for t in range(MC):
      p_hat.append(net.predict(X))
    p_hat = np.array(p_hat)
    # print("P hat shape: ", p_hat.shape)
    # Mean over MC samples (mean over rows)
    mean_probs_over_draw = np.mean(p_hat, axis=0)
    #argmax over the columns 
    predictions_uncertainty = np.argmax(mean_probs_over_draw, axis=1)

    aleatoric = np.mean(p_hat * (1 - p_hat), axis=0)
    epistemic = np.mean(p_hat ** 2, axis=0) - np.mean(p_hat, axis=0) ** 2

    uncertainties_among_labels = epistemic + aleatoric

    predicted_class_variances = np.asarray([uncertainty[prediction] for prediction, uncertainty in
                                            zip(predictions_uncertainty, uncertainties_among_labels)])

    return predictions_uncertainty, predicted_class_variances
    
def compute_uncertainties_softmax(net, X):
    # Softmax std per single predictions --> Softmax uncertainty
    predictions = net.predict(X)
    # prediction per samples are column vector
    # we want to compute the standard deviation for each row -> axis=1
    std_predictions = np.std(predictions, axis=1)
    return std_predictions
