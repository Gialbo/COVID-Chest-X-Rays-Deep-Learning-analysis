# COVID 19 Chest X-Rays Deep Learning analysis
Comparison of different segmentation and synthetic data generation methods applied to chest X Rays from COVID-19 patients. We plan to compare different methods such as UNET, autoencoders, GAN, colorization techniques. \
Final project code for the course "Bioinformatics", A.Y. 2020/2021. \
<img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/images/samples.png"> 


# Project Structure

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
* [`rawGAN.py`](./models/rawGAN.py): first trial using a Generative Adversial Network to generate from scratch X-Rays images. With this first trial we combine all together the three classes of *COVID-19 Radiography Database* to find a good set of hyperparameters to use in the following trials;
* [`covidGAN.py`](./models/rawGAN.py):  Generative Adversial Network to generate synthetic images from *AI for COVID* database.
<!-- * [`cGAN.py`](./models/cGAN.py): starting from the rawGAN, we added to the model to ability to distinguish between the tree different classes. This type is called Conditional GAN; -->


## [`Experiments`](./experiments)
* [`inceptionNet.ipynb`](./experiments/inceptionNet.ipynb): notebook reporting the experiment using the inceptionNet model.
* [`rawGAN.ipynb`](./experiments/rawGAN.ipynb): notebook reporting the experiment using the rawGAN model.
* [`covidGAN.ipynb`](./experiments/rawGAN.ipynb): notebook reporting the experiment using the covidGAN model.


## [`Images`](./images)
The folder containing the images used for the documentation.

## [`Results`](./results)
### Classification Task: inceptionNet
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNet/InceptionNetLoss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNet/InceptionNetAccuracy.png" width="400">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNet/InceptionNetPrecision.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNet/InceptionNetRecall.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/inceptionNet/InceptionNetConfMatrix.png">
 </p>


### Synthetic images generation: covidGAN
<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/covidGAN/covidGAN-faster.gif" width="40%">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidGAN/accuracy.png" width="400">
</p>
 
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidGAN/loss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/covidGAN/lossCloseUp.png" width="400">
</p>

# References
[M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.](https://arxiv.org/ftp/arxiv/papers/2003/2003.13145.pdf)

[AIforCOVID: predicting the clinical outcomes in patients with COVID-19 applying AI to chest-X-rays. An Italian multicenter study.](http://arxiv.org/abs/2012.06531)

[Kora Venu, Sagar and Ravula, Sridhar, "Evaluation of Deep Convolutional Generative Adversarial Networks for data augmentation of chest X-ray images", Future Internet, MDPI AG, 2020](https://arxiv.org/pdf/2009.01181.pdf)

[M. Mirza, S. Osindero, "Conditional Generative Adversarial Nets", 2014](https://arxiv.org/pdf/1411.1784.pdf)
