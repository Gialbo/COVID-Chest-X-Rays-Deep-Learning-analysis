"""
Create gif file from image directory
Bioinformatics, Politecnico di Torino
Authors: Gilberto Manunza, Silvia Giammarinaro
"""

import imageio
import glob

def images_to_gif(img_folder, gif_path):

# Create a gif with generated images from GAN 
# tutorial found here: https://www.tensorflow.org/tutorials/generative/dcgan#create_a_gif

#path images: '/content/drive/MyDrive/BIOINF/images_GAN/one-class/*.png'
#gif path: '/content/drive/MyDrive/BIOINF/images_GAN/one-class/gan-one-class.gif'

  with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
    filenames = glob.glob(img_folder)
    filenames = sorted(filenames)
    for filename in filenames:
      image = imageio.imread(filename)
      writer.append_data(image)

