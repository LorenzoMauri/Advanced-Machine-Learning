import tensorflow as tf
import json
import os
from google.colab import drive
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from time import time
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pylab as plt
import numpy as np
import os
import pathlib
from shutil import copyfile
import tarfile

print('TF version:', tf.__version__)
print('Hub version:', hub.__version__)
print('GPU: ', tf.config.list_physical_devices('GPU'))


#################################
# clone or pull repository github 
#################################

if not os.path.isdir('/content/Advanced-Machine-Learning') : 
  os.system('git clone https://github.com/LorenzoMauri/Advanced-Machine-Learning')
else :
  os.system('git pull') 

os.chdir('/content/Advanced-Machine-Learning/')

#################################
import preprocessing
#################################

preprocess = preprocessing.Preprocessing(
    label_mode = 'int',
    class_names = ['0','90','180','270'],
    color_mode = 'rgb',
    batch_size = 256,
    image_size = (224, 224), # BiT image size > 96 x 96 px # VGG16 (224, 224) # Xception (299, 299)
    seed = 1234,
    validation_split = 0.2,
    interpolation = 'bicubic',
    num_classes = 4,
    labels = 'inferred'
)

############################
# config file 
############################

pathDirectories, root_dir, model_dir, data_dir, remote_indoor_dir, remote_sun_dir, local_indoor_dir, local_sun_dir, indoor_train_dir, indoor_test_dir, sun_train_dir, sun_test_dir = preprocess.getPaths()


preprocess.createFolder(local_indoor_dir)

preprocess.transferDataFromGoogleDrive(destinationDirectory=local_indoor_dir + '/RotatedImages.tar',
                                       currentDirectory=remote_indoor_dir + '/indoorCVPR_09/RotatedImages.tar',
                                       localDirectory=local_indoor_dir)

preprocess.transferDataFromGoogleDrive(destinationDirectory=local_indoor_dir + '/RotatedTestImages.tar',
                                       currentDirectory=remote_indoor_dir + '/indoorCVPR_09/RotatedTestImages.tar',
                                       localDirectory=local_indoor_dir)

preprocess.createFolder(local_sun_dir)

preprocess.transferDataFromGoogleDrive(destinationDirectory=local_sun_dir + '/RotatedImages_224.tar',
                            currentDirectory = remote_sun_dir + '/RotatedImages_224.tar',
                            localDirectory = local_sun_dir)

preprocess.transferDataFromGoogleDrive(currentDirectory = remote_sun_dir + '/RotatedTestImages_224.tar', 
                            destinationDirectory=local_sun_dir + '/RotatedTestImages_224.tar',
                            localDirectory=local_sun_dir)

indoor_train_ds, indoor_validation_ds, indoor_test_ds = preprocess.dataLoader(indoor_train_dir,indoor_test_dir)

sun_train_ds, sun_validation_ds, sun_test_ds = preprocess.dataLoader(sun_train_dir, sun_test_dir) 

# Concatenate indoorCVPR09 and SUN397 partition 01
train_ds = indoor_train_ds.concatenate(sun_train_ds)
validation_ds = indoor_validation_ds.concatenate(sun_validation_ds)
test_ds = indoor_test_ds.concatenate(sun_test_ds)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
validation_ds = validation_ds.prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)
