# COVID 19 Chest X-Rays Deep Learning analysis
Comparison of different segmentation and synthetic data generation methods applied to chest X Rays from COVID-19 patients. We plan to compare different methods such as UNET, autoencoders, GAN, colorization techniques. \
Final project code for the course "Bioinformatics", A.Y. 2020/2021. \
<img src="https://raw.githubusercontent.com/Gialbo/COVID-Chest-X-Rays-Deep-Learning-analysis/main/results/gan-one-class.gif" width="40%">


# Project Structure

##  [`Data`](./data)
The dataset contains X-rays images from different patients with different patologies: there are 1200 COVID-19 positive images, 1341 normal images, and 1345 viral pneumonia images. The dataset can be downloaded from [`here`](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database).
From this dataset we applied some preprocessing techniques in order to have the data ready for our experiments.
* [`train_test_split.py`](./data/train_test_split.py): create a new folder divided in two subfolders: train and test. These folders are needed to define the ImageDataGenerator for training.
* [`resize_images.py`](./data/resize_images.py): resize all the images in the dataset to 224x224 pizels.

## [`Models`](./models)
...

## [`Experiments`](./experiments)
...

## [`Results`](./results)
* [`gan-one-class.gif`](./results/gan-one-class.gif) shows how the generator is learning to generate true X-rays images. We keep the latent space fixed and we take the output of the generator every 10 epochs for a total of 400 epochs. In this case we are using all the classes merged together, so one class is used.



# References
M.E.H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M.A. Kadir, Z.B. Mahbub, K.R. Islam, M.S. Khan, A. Iqbal, N. Al-Emadi, M.B.I. Reaz, M. T. Islam, “Can AI help in screening Viral and COVID-19 pneumonia?” IEEE Access, Vol. 8, 2020, pp. 132665 - 132676.
