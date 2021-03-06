# COVID 19 Chest X-Rays Deep Learning analysis
Comparison of different segmentation and synthetic data generation methods applied to chest X Rays from COVID-19 patients. We plan to compare different methods such as UNET, autoencoders, GAN, colorization techniques. \
Final project code for the course "Bioinformatics", A.Y. 2020/2021. \
<img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/images/samples.png"> 


# Project Structure

##  [`Data`](./data)
The dataset contains X-rays images from different patients with different patologies: there are 1200 COVID-19 positive images, 1341 normal images, and 1345 viral pneumonia images.

<p align="center">
  <img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/images/dataDistribution.png"> 
</p>

The dataset can be downloaded from [`here`](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).
From this dataset we applied some preprocessing techniques in order to have the data ready for our experiments. 
* [`train_test_split.py`](./data/train_test_split.py): create a new folder divided in two subfolders: train and test. These folders are needed to define the ImageDataGenerator for training.
* [`resize_images.py`](./data/resize_images.py): resize all the images in the dataset to 224x224 pixels.
* [`load_data.py`](./data/load_data.py): load data from path and return the datasets (train, validation, test)
* [`images_to_gif.py`](./data/images_to_gif.py): create a gif with generated images from GAN 

After these passages, we are ready to train our models. Our final dataset can be downloaded [`here`](https://drive.google.com/drive/folders/1-7se3aMXMXtDF89ALV07pru3kELmWTTo?usp=sharing) and it was created using the version 2 of the original dataset.

## [`Models`](./models)
* [`inceptionNet.py`](./models/inceptionNet.py): CNN model used for the classification task. We first loaded the inceptionV3 model with imagenet weight and added more layers at the top. We do not freeze any layer, so during training the preloaded weights from Imagenet will be updated;
* [`rawGAN.py`](./models/rawGAN.py): first trial using a Generative Adversial Network to generate from scratch X-Rays images. With this first trial we combine all together the three classes of our dataset to find a good set of hyperparameters to use in the following trials;
* [`cGAN.py`](./models/cGAN.py): starting from the rawGAN, we added to the model to ability to distinguish between the tree different classes. This type is called Conditional GAN;


## [`Experiments`](./experiments)
* [`inceptionNet.ipynb`](./experiments/inceptionNet.ipynb): notebook reporting the experiment using the inceptionNet model.
* [`rawGAN.ipynb`](./experiments/rawGAN.ipynb): notebook reporting the experiment using the rawGAN model.

## [`Images`](./images)
The folder containing the images used for the documentation.

## [`Results`](./results)
### Classification Task: inceptionNet
<p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/InceptionNetLoss.png" width="400">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/InceptionNetAccuracy.png" width="400">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/InceptionNetPrecision.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/InceptionNetRecall.png">
 </p>
 
 <p align="center">
  <img src="https://github.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/blob/main/results/InceptionNetConfMatrix.png">
 </p>

### Synthetic images generation: rawGAN
* [`gan-one-class.gif`](./results/gan-one-class.gif) shows how the generator is learning to generate true X-rays images. We keep the latent space fixed and we take the output of the generator every 10 epochs for a total of 400 epochs. In this case we are using all the classes merged together, so one class is used. Here we can see the learning process of the generator. 
<img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/gan-one-class.gif" width="40%">

# References
[M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.](https://arxiv.org/ftp/arxiv/papers/2003/2003.13145.pdf)

[Kora Venu, Sagar and Ravula, Sridhar, "Evaluation of Deep Convolutional Generative Adversarial Networks for data augmentation of chest X-ray images", Future Internet, MDPI AG, 2020](https://arxiv.org/pdf/2009.01181.pdf)

[M. Mirza, S. Osindero, "Conditional Generative Adversarial Nets", 2014](https://arxiv.org/pdf/1411.1784.pdf)
