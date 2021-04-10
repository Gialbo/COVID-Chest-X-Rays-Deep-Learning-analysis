# COVID 19 Chest X-Rays Deep Learning analysis
Comparison of different segmentation and synthetic data generation methods applied to chest X Rays from COVID-19 patients. We plan to compare different methods such as UNET, autoencoders, GAN, colorization techniques. \
Final project code for the course "Bioinformatics", A.Y. 2020/2021. 
# Table of Contents
1. [Data](#data)
2. [Models](#models)
3. [Experiments](#experiments)
4. [Results](#results)
5. [References](#references)
<img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/images/samples.png"> 


##  [`Data`](./data)
We used two different datasets: COVID-19 RaioGraphy Database and AI for COVID. Further information about the dataset are reported below.
From these datasets we applied some preprocessing techniques in order to have the data ready for our experiments. 
* [`train_test_split.py`](./data/train_test_split.py): create a new folder divided in two subfolders: train and test. These folders are needed to define the ImageDataGenerator for training.
* [`resize_images.py`](./data/resize_images.py): resize all the images in the dataset to 224x224 pixels.
* [`load_data.py`](./data/load_data.py): load data from path and return the datasets (train, validation, test)
* [`images_to_gif.py`](./data/images_to_gif.py): create a gif with generated images from GAN 

After these passages, we are ready to train our models. Our final datasets can be downloaded here: [`COVID-19 Radiography Database`](https://drive.google.com/drive/folders/1-7se3aMXMXtDF89ALV07pru3kELmWTTo?usp=sharing), [`AI for COVID`](https://drive.google.com/drive/u/0/folders/150mo0iE72Fs4j7dYyeNxGSJyAdDLbMpx)

#### [`COVID-19 Radiography Database`](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
The dataset contains X-rays images from different patients with different patologies: there are 1027 COVID-19 positive images, 1206 normal images, and 1210 viral pneumonia images.

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/images/dataDistribution.png"> 
</p>

#### [`AI for COVID`](https://aiforcovid.radiomica.it/)
The dataset is provided by CDI (Centro Diagnostico Italiano) and it contains X-rays from patients with COVID-19. We have a total of 696 images. 


## [`Models`](./models)
* [`inceptionNet.py`](./models/inceptionNet.py): CNN model used for the classification task on *COVID-19 Radiography Database*. We first loaded the inceptionV3 model with imagenet weight and added more layers at the top. We do not freeze any layer, so during training the preloaded weights from Imagenet will be updated;
* [`inceptionV3MCD.py`](./models/inceptionV3MCD.py): modified version of InceptionV3 implemented in Keras. To each block a dropout layer is added at the end of it. The rate of the dropout layer can be passed calling the function;
* [`inceptionNetMCD.py`](./models/inceptionNetMCD.py): Monte Carlo Dropout inceptionNet. The main difference are the following: the inceptionNetV3MCD is used and dropout layers are added after every fully connected layer. 

* [`covidGAN.py`](./models/rawGAN.py):  Generative Adversial Network to generate synthetic images from *AI for COVID* database.
* [`cGAN.py`](./models/cGAN.py): starting from the covidGAN, we added to the model to ability to distinguish between the tree different classes. This architecture is called Conditional GAN.  
<!-- [`rawGAN.py`](./models/rawGAN.py): first trial using a Generative Adversial Network to generate from scratch X-Rays images. With this first trial we combine all together the three classes of *COVID-19 Radiography Database* to find a good set of hyperparameters to use in the following trials -->



## [`Experiments`](./experiments)
* [`inceptionNet.ipynb`](./experiments/inceptionNet.ipynb): notebook reporting the experiment using the inceptionNet model.
* [`inceptionNetMCD.ipynb`](./experiments/inceptionNetMCD.ipynb): notebook reporting the experiment using the modified version of the inceptionNet model with Monte Carlo dropout.
* [`covidGAN.ipynb`](./experiments/rawGAN.ipynb): notebook reporting the experiment using the covidGAN model.
* [`cGAN.ipynb`](./experiments/cGAN.ipynb): notebook reporting the experiment using the cGAN model. 
<!-- * [`rawGAN.ipynb`](./experiments/rawGAN.ipynb): notebook reporting the experiment using the rawGAN model. -->


## [`Results`](./results)
### Classification Task: inceptionNet
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

### Classification Task: deterministic inceptionNet vs Monte Carlo Dropout inceptionNet

In the table below are reported the results on the classification task. The recall and precision values are reported for the classes in the following order: covid-19, normal and viral pneumonia.


| Model                         |    Accuracy   | Loss    | Recall                 | Precision             |
| --------------------------    | ------------- | --------| -----------------------| ----------------------|
| inceptionNet (deterministic)  |     0.9347    |  0.3558 | 0.9739; 0.9776; 0.8582 | 1; 0.8618; 0.9664     |
| inceptionNetMCD               |     0.9191    |  0.2434 | 1; 0.8358; 0.9254      | 0.9055; 0.9655; 0.8921|

 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNet/confMatrix.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNetMCD/confMatrix.png" width="400">
 </p>


To compare the uncertainty of both networks, the following strategies are used:
* inceptionNet (deterministic): in the deterministic net, the uncertainty of a prediction can be computed looking at the output softmax vector. The uncertainty of the whole test set is expressed as the standard deviation of the softmax vector for every image.
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
 

### Synthetic images generation: covidGAN
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidGAN/imagesCompared.png">
 </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/covidGAN/covidGAN-faster.gif" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidGAN/accuracy.png" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidGAN/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidGAN/lossCloseUp.png" width="400">
</p>

### Synthetic images generation: cGAN
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN/imagesCompared.png">
 </p>

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/cGAN/cGAN.gif" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN/accuracy.png" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/cGAN/lossCloseUp.png" width="400">
</p>


# References
[Can AI help in screening Viral and COVID-19 pneumonia? ( Chowdhury et al., IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.)](https://arxiv.org/ftp/arxiv/papers/2003/2003.13145.pdf)

[AIforCOVID: predicting the clinical outcomes in patients with COVID-19 applying AI to chest-X-rays. An Italian multicenter study.](http://arxiv.org/abs/2012.06531)

[Uncertainty quantification using Bayesian neural networks in classification: Application to biomedical image segmentation (Kwon et al., 2019)](https://openreview.net/pdf?id=Sk_P2Q9sG)

[Evaluation of Deep Convolutional Generative Adversarial Networks for data augmentation of chest X-ray images (Venu et al., Future Internet, MDPI AG, 2020)](https://arxiv.org/pdf/2009.01181.pdf)

[Conditional Generative Adversarial Nets (Mirza et al. 2014)](https://arxiv.org/pdf/1411.1784.pdf)
