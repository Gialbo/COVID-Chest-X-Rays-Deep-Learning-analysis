import os, shutil

def train_test_split(original_dataset_dir, base_dir):
    # Create a new folder divided in two subfolders: train and test 
    # These folders are needed to define the ImageDataGenerator for training

    # Input:
    # original_dataset_dir: the path of the original folder containing all the data
    # base_dir: the path of the new folder
    
    #original_dataset_dir="/content/drive/MyDrive/BIOINF/covid-project/COVID-19 Radiography Database"
    #base_dir="/content/drive/MyDrive/BIOINF/covid-project/COVID-19-xrays"

    path, dirs, files = next(os.walk(original_dataset_dir + "/COVID-19"))
    covid_19 = len(files)
    print("N. samples COVID ", covid_19)

    path, dirs, files = next(os.walk(original_dataset_dir + "/NORMAL"))
    normal = len(files)
    print("N. samples normal ", normal)

    path, dirs, files = next(os.walk(original_dataset_dir + "/Viral Pneumonia"))
    pneumonia = len(files)
    print("N. samples pneumonia ", pneumonia)

    os.mkdir(base_dir)

    # Directory for the training splits
    train_dir = os.path.join(base_dir, 'train')
    os.mkdir(train_dir)

    # Directory for the test splits
    test_dir = os.path.join(base_dir, 'test')
    os.mkdir(test_dir)

    # Create train directories
    train_covid_dir = os.path.join(train_dir, 'covid-19')
    os.mkdir(train_covid_dir)
    train_normal_dir = os.path.join(train_dir, 'normal')
    os.mkdir(train_normal_dir)
    train_pneumonia_dir = os.path.join(train_dir, 'viral-pneumonia')
    os.mkdir(train_pneumonia_dir)

    # Create test directories
    test_covid_dir = os.path.join(test_dir, 'covid-19')
    os.mkdir(test_covid_dir)
    test_normal_dir = os.path.join(test_dir, 'normal')
    os.mkdir(test_normal_dir)
    test_pneumonia_dir = os.path.join(test_dir, 'viral-pneumonia')
    os.mkdir(test_pneumonia_dir)

    # copy COVID samples into train folder
    fnames = ['COVID-19 ({}).png'.format(i) for i in range(1, 1028)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'COVID-19', fname)
        dst = os.path.join(train_covid_dir, fname)
        shutil.copyfile(src, dst)

    # copy COVID samples into train folder
    fnames = ['COVID-19 ({}).png'.format(i) for i in range(1028, 1143)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'COVID-19', fname)
        dst = os.path.join(test_covid_dir, fname)
        shutil.copyfile(src, dst)

    # copy NORMAL samples into train folder
    fnames = ['NORMAL ({}).png'.format(i) for i in range(1, 1207)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'NORMAL', fname)
        dst = os.path.join(train_normal_dir, fname)
        shutil.copyfile(src, dst)

    # copy NORMAL samples into train folder
    fnames = ['NORMAL ({}).png'.format(i) for i in range(1207, 1341)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'NORMAL', fname)
        dst = os.path.join(test_normal_dir, fname)
        shutil.copyfile(src, dst)

    # copy Viral Pneumonia samples into train folder
    fnames = ['Viral Pneumonia ({}).png'.format(i) for i in range(1, 1211)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'Viral Pneumonia', fname)
        dst = os.path.join(train_pneumonia_dir, fname)
        shutil.copyfile(src, dst)

    # copy Viral Pneumonia samples into train folder
    fnames = ['Viral Pneumonia ({}).png'.format(i) for i in range(1211, 1345)]
    for fname in fnames:
        src = os.path.join(original_dataset_dir, 'Viral Pneumonia', fname)
        dst = os.path.join(test_pneumonia_dir, fname)
        shutil.copyfile(src, dst)

    print("Samples COVID train: ", len(os.listdir(train_covid_dir)))
    print("Samples COVID test: ", len(os.listdir(test_covid_dir)))
    print("Samples NORMAL train: ", len(os.listdir(train_normal_dir)))
    print("Samples NORMAL test: ", len(os.listdir(test_normal_dir)))
    print("Samples Viral Pneumonia train: ", len(os.listdir(train_pneumonia_dir)))
    print("Samples Viral Pneumonia test: ", len(os.listdir(test_pneumonia_dir)))

