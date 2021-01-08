import os, sys
from PIL import Image

## !unzip "/content/drive/MyDrive/BIOINF/covid-project/COVID-19-xrays" -d "/content/drive/MyDrive/BIOINF/covid-project/COVID-19-xrays-resized"
## dir_path = "/content/drive/MyDrive/BIOINF/covid-project/COVID-19-xrays-resized/COVID-19-xrays"

def resize_images(dir_path):
    # Resize all the images in the dataset to 224x224 pizels
    
    # Input:
    # dir_path: directory containing all the images to resize

    resize_all(dir_path)
# dir_path = "/content/drive/MyDrive/BIOINF/covid-project/COVID-19-xrays-resized/COVID-19-xrays"

def resize_im(path):
    if os.path.isfile(path):
        im = Image.open(path).resize((224,224), Image.ANTIALIAS)
        parent_dir = os.path.dirname(path)
        img_name = os.path.basename(path).split('.')[0]
        im.save(os.path.join(parent_dir, img_name + '.png'), 'PNG', quality=90)

def resize_all(mydir):
    for subdir , _ , fileList in os.walk(mydir):
        for f in fileList:
            try:
                full_path = os.path.join(subdir,f)
                resize_im(full_path)
            except Exception as e:
                print('Unable to resize %s. Skipping.' % full_path)

