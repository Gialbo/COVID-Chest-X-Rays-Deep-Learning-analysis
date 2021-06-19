# COVID 19 Chest X-Rays Deep Learning analysis
Comparison of different GAN-based synthetic data generation methods applied to chest X-Rays from COVID-19, Viral Pneumonia and Normal patients. 
Final project for the course "Bioinformatics", A.Y. 2020/2021. 
# Table of Contents
1. [Data](#data)
2. [Models](#models)
3. [Experiments](#experiments)
4. [Tools](#tools)
5. [Generation Results](#generation-results)
6. [Classification Results](#classification-results)
7. [Generative Classification Results](#generative-classification-results)
8. [Frechet Inception Distance Results](#frechet-inception-distance-results)
9. [References](#references)
<img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/images/samples.png"> 
<center><em>Real images coming from the dataset</em></center>


##  [`Data`](./data)
We used the COVID-19 RadioGraphy Database. Further information about the dataset are reported below.
Starting From the dataset we applied some preprocessing techniques in order to have the data ready for our experiments. 
* [`train_test_split.py`](./data/train_test_split.py): create a new folder divided in two subfolders: train and test.
* [`resize_images.py`](./data/resize_images.py): resize all the images in the dataset to 224x224 pixels.

After these passages, we are ready to train our models. Our final dataset can be downloaded here: [`COVID-19 Radiography Database`](https://drive.google.com/drive/folders/1-7se3aMXMXtDF89ALV07pru3kELmWTTo?usp=sharing).

#### [`COVID-19 Radiography Database`](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
The dataset contains X-rays images from different patients with different patologies: there are 1027 COVID-19 positive images, 1206 normal images, and 1210 viral pneumonia images.

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/images/dataDistribution.png"> 
</p>



## [`Models`](./models)
* [`inceptionNet.py`](./models/inceptionNet.py): CNN model used for the classification task on *COVID-19 Radiography Database*. We first loaded the inceptionV3 model with imagenet weight and added more layers at the top. We do not freeze any layer, so during training the preloaded weights from Imagenet will be updated;
* [`inceptionV3MCD.py`](./models/inceptionV3MCD.py): modified version of InceptionV3 implemented in Keras. To each block a dropout layer is added at the end of it. The rate of the dropout layer can be passed calling the function;
* [`inceptionNetMCD.py`](./models/inceptionNetMCD.py): Monte Carlo Dropout inceptionNet. The main difference are the following: the inceptionNetV3MCD is used and dropout layers are added after every fully connected layer. 
* [`covidGAN.py`](./models/covidGAN.py):  Generative Adversial Network to generate synthetic COVID-19 x-rays samples  from the *COVID-19 Radiography Database* database.
* [`covidUnetCGAN`](./models/unetCGAN.py): Particular version of a classical Generative Adversarial Network in which the discriminator is substituted with a Unet net. The Uney will becomposed by an encoder and a decoder. If we shrink to $1$ the output of the encoder we obtain the oputput of a classical discriminator net. But additionally we re decode the image and we tell the network to maximize the pixel-wise cross entropy of the decoded output. This means that in output of the decode we will have a map in which each pixel tells us in a grayscale rapresentation how much confident the network is for that pixel of the image being true. A value for a pixel close the $1$ (white) means that for that pixel the networ is sure of the image being real and viceversa for a value closs to $0$ (black) the network is sure for the image of being fake.
* [`cGAN.py`](./models/cGAN.py): starting from the covidGAN, we added random labels as input of the generator to generate images according to a given class. This architecture is called Conditional GAN. Futhermore, to make the training more stable, we added residual connections in the generator.
* [`unetCGAN.py`](./models/unetCGAN.py): to do...
* [`ACCGAN.py`](./models/ACCGAN.py): to do...
* [`cGAN_Uncertainty.py`](./models/cGAN_Uncertainty.py): to do...
* [`ACCGAN_Uncertainty.py`](./models/ACCGAN_Uncertainty.py): to do...
* [`GenerativeClassification.py`](./models/GenerativeClassification.py): to do...




## [`Experiments`](./experiments)
* [`dataExploration.ipynb`](./experiments/dataExploration.ipynb): visualize the images and the class distribution.
* [`inceptionNet.ipynb`](./experiments/inceptionNet.ipynb): notebook reporting the experiment using the inceptionNet model.
* [`inceptionNetMCD.ipynb`](./experiments/inceptionNetMCD.ipynb): notebook reporting the experiment using the modified version of the inceptionNet model with Monte Carlo dropout.
* [`covidGAN.ipynb`](./experiments/covidGAN.ipynb): notebook reporting the experiment using the covidGAN model.
* [`covidUnetGAN.ipynb`](./experiments/covidUnetGAN.ipynb): notebook reporting the experiment using the covidUnetGAN model.
* [`cGAN.ipynb`](./experiments/cGAN.ipynb): notebook reporting the experiment using the cGAN and cGAN_Uncertainty model.
* [`unetcGAN.ipynb`](./experiments/cGAN.ipynb): notebook reporting the experiment using the unetcGAN model.
* [`AC-CGAN.ipynb`](./experiments/AC-CGAN.ipynb): notebook reporting the experiment using the ACCGAN and ACCGAN_Uncertainty model.
* [`compute_FID.ipynb`](./experiments/compute_FID.ipynb):  notebook reporting the experiment to compute FID for all the generated images by each model.
* [`InceptionGenerativeClassification.ipynb`](./experiments/InceptionGenerativeClassification.ipynb): to do...

##  [`Tools`](./tools)
* [`FID.py`](./tools/FID.py): class used to compute FID (Frechet Inception Distance);
* [`images_to_gif.py`](./tools/images_to_gif.py): create a gif from the generated images by the model;
* [`plotter.py`](./tools/plotter.py): utility functions for plotting results;
* [`uncertainty.py`](./tools/uncertainty.py): utility functions to compute uncertainty for deterministic and Monte Carlo Dropout models;
* [`XRaysDataset.py`](./tools/XRaysDataset.py): load and preprocess data from given directory. This tools permits to set the final image size needed to pass the images to the model and sets up the prefetching of thedataset for increased performances.

 
## Generation Results

### covidGAN
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidGAN/imagesCompared.png">
 </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/covidGAN/covidGAN.gif" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidGAN/accuracy.png" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidGAN/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidGAN/lossCloseUp.png" width="400">
</p>

### covidUnetGAN

 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidUnetGAN/imagesCompared.png">
 </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/covidUnetGAN/covidUnetGAN_gen.gif" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidUnetGAN/covidUnetGAN_dec.gif" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidUnetGAN/accuracy.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidUnetGAN/loss.png" width="400">
</p>


### cGAN

 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN/imagesCompared.png">
 </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/cGAN/cGAN.gif" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN/accuracy.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN/loss.png" width="400">
</p>


### cGAN with uncertainty (min)

 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN-unc-min/imagesCompared.png">
 </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/cGAN-unc-min/cGAN_unc_min.gif" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN-unc-min/accuracy.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN-unc-min/loss.png" width="400">
</p>

### cGAN with uncertainty (max)

 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN-unc-max/imagesCompared.png">
 </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/cGAN-unc-max/cGAN_unc_max.gif" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN-unc-max/accuracy.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN-unc-max/loss.png" width="400">
</p>


### unetCGAN

 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/unetCGAN/imagesCompared.png">
 </p> 

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/unetCGAN/unetCGAN_gen.gif" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/unetCGAN/unetCGAN_dec.gif" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/unetCGAN/accuracy.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/unetCGAN/loss.png" width="400">
</p>

### AC-CGAN
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/AC-CGAN/imagesCompared.png">
 </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/AC-CGAN/AC-CGAN.gif" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/AC-CGAN/accuracy.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/AC-CGAN/loss.png" width="400">
</p>


### AC-CGAN with uncertainty (min)
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/AC-CGAN-unc-min/imagesCompared.png">
 </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/AC-CGAN-unc-min/AC-CGAN_unc_min.gif" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/AC-CGAN-unc-min/accuracy.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/AC-CGAN-unc-min/loss.png" width="400">
</p>

### AC-CGAN with uncertainty (max)

<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/AC-CGAN-unc-max/imagesCompared.png">
 </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/AC-CGAN-unc-max/AC-CGAN_unc_max.gif" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/AC-CGAN-unc-max/accuracy.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/AC-CGAN-unc-max/loss.png" width="400">
</p>

## Classification Results

### Classification Task: deterministic inceptionNet
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNet/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNet/accuracy.png" width="400">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNet/precision.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNet/recall.png">
 </p>
 

### Classification Task: Monte Carlo Dropout inceptionNet
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNetMCD/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNetMCD/accuracy.png" width="400">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNetMCD/precision.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNetMCD/recall.png">
 </p>

### Classification Task: deterministic inceptionNet vs Monte Carlo Dropout inceptionNet

In the table below are reported the overall results on the classification task. The results of the deterministic model are obtained running the training configuration for five times. Instead for the Monte Carlo Dropout model the evaluation configuration is run five times due to the active dropout layers.
<!--
| Model                         |    Accuracy   | Loss    | Recall                 | Precision             |
| --------------------------    | ------------- | --------| -----------------------| ----------------------|
| inceptionNet (deterministic)  |     0.9347    |  0.3558 | 0.9739; 0.9776; 0.8582 | 1; 0.8618; 0.9664     |
| inceptionNetMCD               |     0.9191    |  0.2434 | 1; 0.8358; 0.9254      | 0.9055; 0.9655; 0.8921| -->


  | Model                        | Accuracy        | Loss            |
  |------------------------------|-----------------|-----------------|
  | inceptionNet (deterministic) |**0.944 ± 0.026**|**0.420 ± 0.274**|
  | inceptionNetMCD              | 0.907 ± 0.007   | 0.326 ± 0.032   |

  | Model                         | Recall, Covid-19  | Recall, Normal  | Recall, Viral Pneumonia |
  |-------------------------------|-------------------|-----------------|-------------------------|
  | inceptionNet  (deterministic) | 0.939 ± 0.066     |**0.966 ± 0.025**| **0.928 ± 0.046**       |
  | inceptionNetMCD               |**0.974 ± 0.007**  | 0.894 ± 0.009   | 0.861 ± 0.010           |

  | Model                         | Precision, Covid-19  | Precision, Normal | Precision, Viral Pneumonia |
  |-------------------------------|----------------------|-------------------|----------------------------|
  | inceptionNet  (deterministic) | **1.000 ± 0.000**    | 0.917 ± 0.060     | **0.936 ± 0.020**          |
  | inceptionNetMCD               | 0.933 ± 0.009        | **0.922 ± 0.011** | 0.878 ± 0.010              |


 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNet/confMatrix.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNetMCD/confMatrix.png" width="400">
 </p>

### Visualize the uncertainty: deterministic inceptionNet vs Monte Carlo Dropout inceptionNet


To compare the uncertainty of both networks, the following strategies are used:
* inceptionNet (deterministic): in the deterministic net, the uncertainty of a prediction can be computed from the output softmax vector. The uncertainty of the whole test set is expressed as the standard deviation of the softmax vector for every image.
* inceptionNetMCD: in the Monte Carlo Dropout setting, we compute for n times the predictions of the net (n is set to 100). In this case the uncertainty is the following:
<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/images/uncertainty_formula.png"  width="300"> 
</p>
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/images/uncertainty_formula_desc.png" width="450" >



Once we compute the uncertainties for both networks, the experiments can be easily compared using barplots. A low level of uncertainty means the network is sure about the prediction and viceversa. The deterministic net (left) behave as expected, even if the accuracy is high on the test set, most of the prediction are highly unsure. Instead the Monte Carlo Dropout network (right) is more confident in the predictions. The behaviour can be fully exploited filtering the prediction in correct and wrong.
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNetMCD/allPredictions.png">
 </p>
 
Plotting only correct or wrong predictions shows how the Monte Carlo Dropout network is working in the desired way. For correct prediction, most of them have a lower uncertainty. Instead, for the wrong ones, the uncertainty is very high, even if it is lower than the max value found with the deterministic model.
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNetMCD/correctPredictions.png">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNetMCD/wrongPredictions.png">
 </p>
 

## Generative Classification Results

description to do...


  | Model                        | Accuracy         | Loss             |
  |------------------------------|------------------|------------------|
  | cGAN                         | 0.959 ± 0.007    | 0.203 ± 0.046    |
  | cGAN + uncertainty (min)     |**0.968 ± 0.008** |**0.119 ± 0.042** |
  | cGAN + uncertainty (max)     | 0.965 ± 0.004    | 0.167 ± 0.054    |
  | AC-CGAN                      | 0.948 ± 0.016    | 0.246 ± 0.092    |
  | AC-CGAN + uncertainty (min)  | 0.950 ± 0.008    | 0.180 ± 0.061    |
  | AC-CGAN + uncertainty (max)  | 0.956 ± 0.011    | 0.206 ± 0.063    |

  | Model                         | Recall, Covid-19  | Recall, Normal  | Recall, Viral Pneumonia |
  |-------------------------------|-------------------|-----------------|-------------------------|
  | cGAN                          | 0.974 ± 0.012     | 0.930 ± 0.019   |  0.973 ± 0.017          |
  | cGAN + uncertainty (min)      | **0.981 ± 0.007** |**0.960 ± 0.023**|  0.966 ± 0.010          |
  | cGAN + uncertainty (max)      | 0.979 ± 0.009     | 0.951 ± 0.013   |  0.967 ± 0.012          |
  | AC-CGAN                       | 0.979 ± 0.009     | 0.894 ± 0.039   |  **0.975 ± 0.007**      |
  | AC-CGAN + uncertainty (min)   | 0.977 ± 0.017     | 0.921 ± 0.017   |  0.954 ± 0.027          |
  | AC-CGAN + uncertainty (max)   | 0.977 ± 0.009     | 0.922 ± 0.024   |  0.970 ± 0.016          |

  | Model                         | Precision, Covid-19  | Precision, Normal | Precision, Viral Pneumonia |
  |-------------------------------|----------------------|-------------------|----------------------------|
  | cGAN                          | **1.000 ± 0.000**    | **0.972 ± 0.016** |  0.917 ± 0.023             |
  | cGAN + uncertainty (min)      | 0.997 ± 0.007        | 0.966 ± 0.009     |  **0.948 ± 0.019**         |
  | cGAN + uncertainty (max)      | **1.000 ± 0.000**    | 0.965 ± 0.014     |  0.937 ± 0.007             |
  | AC-CGAN                       | 0.995 ± 0.004        | 0.971 ± 0.014     |  0.897 ± 0.037             |
  | AC-CGAN + uncertainty (min)   | 0.995 ± 0.004        | 0.955 ± 0.028     |  0.912 ± 0.011             |
  | AC-CGAN + uncertainty (max)   | **1.000 ± 0.000**    | 0.961 ± 0.023     |  0.921 ± 0.029             |

### cGAN 

<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN/accuracy.png" width="400">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN/precision.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN/recall.png">
 </p>
 
 ### cGAN + uncertainty (min)
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN-unc-min/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN-unc-min/accuracy.png" width="400">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN-unc-min/precision.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN-unc-min/recall.png">
 </p>
 
  ### cGAN + uncertainty (max)
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN-unc-max/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN-unc-max/accuracy.png" width="400">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN-unc-max/precision.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/cGAN-unc-max/recall.png">
 </p>

### AC-CGAN 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN/accuracy.png" width="400">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN/precision.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN/recall.png">
 </p>
 

 
  ### AC-CGAN + uncertainty (min)
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN-unc-min/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN-unc-min/accuracy.png" width="400">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN-unc-min/precision.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN-unc-min/recall.png">
 </p>
 
 ### AC-CGAN + uncertainty (max)
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN-unc-max/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN-unc-max/accuracy.png" width="400">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN-unc-max/precision.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/generative-classification/AC-CGAN-unc-max/recall.png">
 </p>
 
## Frechet Inception Distance Results

To measure the quality of the generated images compared to the original ones, we use a technique called Frechet Inception Distance. Given the statistics of the real and the generated images, the distance is computed as an improvment of the Inception Score (IS) in the following way:
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/images/FID.png">
 </p>
 To run the experiments, we create 5 different sets of points from the latent space and/or random labels. 

| Model                         |    FID          | 
| --------------------------    | --------------- | 
| covidGAN                      | 313.84 ± 2.48   |  
| covidUnetGAN                  | 188.48 ± 3.84   |  
| cGAN                          | 80.65  ± 1.27   | 
| cGAN + uncertainty (min)      | 72.68  ± 0.92   |  
| cGAN + uncertainty (max)      |**62.59  ± 0.61**|  
| unetCGAN                      | 89.76  ± 2.03   |  
| AC-CGAN                       | 81.87  ± 1.85   |  
| AC-CGAN + uncertainty (min)   | 89.65  ± 1.45   |  
| AC-CGAN + uncertainty (max)   | 78.54  ± 1.59   |  


# References
[Can AI help in screening Viral and COVID-19 pneumonia? ( Chowdhury et al., IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.)](https://arxiv.org/ftp/arxiv/papers/2003/2003.13145.pdf)

[Generative Adversial Network (Goodfellow et al., 2014)](https://arxiv.org/pdf/1406.2661.pdf)


[A U-Net Based Discriminator for Generative Adversarial Networks (Schonfeld et al., CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Schonfeld_A_U-Net_Based_Discriminator_for_Generative_Adversarial_Networks_CVPR_2020_paper.pdf)

[Conditional Generative Adversarial Nets (Mirza et al., 2014)](https://arxiv.org/pdf/1411.1784.pdf)

[Conditional Image Synthesis with Auxiliary Classifier GANs (Odena et al., 2017)](https://arxiv.org/pdf/1610.09585.pdf)

[Evaluation of Deep Convolutional Generative Adversarial Networks for data augmentation of chest X-ray images (Venu et al., Future Internet, MDPI AG, 2020)](https://arxiv.org/pdf/2009.01181.pdf)

[Uncertainty quantification using Bayesian neural networks in classification: Application to biomedical image segmentation (Kwon et al., 2019)](https://openreview.net/pdf?id=Sk_P2Q9sG)

<!-- [AIforCOVID: predicting the clinical outcomes in patients with COVID-19 applying AI to chest-X-rays. An Italian multicenter study.](http://arxiv.org/abs/2012.06531) -->





