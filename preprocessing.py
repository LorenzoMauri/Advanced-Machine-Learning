# -*- coding: utf-8 -*-
"""
Created on Sat Mar 20 12:31:17 2021

@author: loren
"""

import tensorflow as tf
import json
import os
from google.colab import drive
import tensorflow_hub as hub
import tensorflow_datasets as tfds
from time import time
from PIL import Image
from io import BytesIO
import matplotlib.pylab as plt
import numpy as np
import os
from shutil import copyfile
import tarfile



os.chdir(os.path.dirname(__file__))


class Preprocessing():
    def __init__(self,configFileDict, batch_size,
               label_mode, class_names,
               num_classes, interpolation,
               validation_split, image_size,
               seed, color_mode 
               ) :
      
        self.config =  configFileDict
        
        self.batch_size = batch_size
        self.label_mode = label_mode
        self.class_names = class_names 
        self.num_classes = num_classes 
        self.interpolation = interpolation
        self.validation_split = validation_split
        self.image_size = image_size 
        self.seed = seed
        self.color_mode = color_mode 
        
    def __createFolder(name) :
        if not os.path.exists(name):
          os.mkdir(name)

      
    def dataLoader(self,datasetName,trainDataDirectory,
                   testDataDirectory):
             
    
        # Train
        self.train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            trainDataDirectory,
            labels=self.labels,
            label_mode= self.label_mode,
            class_names= self.class_names,
            color_mode= self.color_mode,
            batch_size= self.batch_size,
            image_size= self.image_size,
            shuffle=True,
            seed=self.seed,
            validation_split=self.validation_split,
            subset='training',
            interpolation=self.interpolation,
            follow_links=False,
        )
    
        # Validation
        validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
            trainDataDirectory,
            labels=self.labels,
            label_mode= self.label_mode,
            class_names= self.class_names,
            color_mode= self.color_mode,
            batch_size= self.batch_size,
            image_size= self.image_size,
            shuffle=True,
            seed=self.seed,
            validation_split=self.validation_split,
            subset='validation',
            interpolation=self.interpolation,
            follow_links=False,
        )
    
        # Test
        test_ds = tf.keras.preprocessing.image_dataset_from_directory(
            testDataDirectory,
            labels= self.labels,
            label_mode= self.label_mode,
            class_names= self.class_names,
            color_mode= self.color_mode,
            batch_size= self.batch_size,
            image_size= self.image_size,
            shuffle=True,
            seed=self.seed,
            validation_split=self.validation_split,
            subset='training',
            interpolation=self.interpolation,
            follow_links=False,
        )
    
        # controls 
        assert isinstance(self.train_ds, tf.data.Dataset)
        print(f"Number of train batches: {tf.data.experimental.cardinality(self.train_ds)}")
        assert isinstance(validation_ds, tf.data.Dataset)
        print(f"Number of validation batches: {tf.data.experimental.cardinality(validation_ds)}")
        assert isinstance(test_ds, tf.data.Dataset)
        print(f"Number of test batches: {tf.data.experimental.cardinality(test_ds)}")
    
        return self.train_ds, validation_ds, test_ds
    
    
    
    def visualizeImages(self,grid=3,figsize = (10,10)) : 
        plt.figure(figsize=figsize)
        for images, labels in self.train_ds.take(1):
          for i in range(grid*grid):
            ax = plt.subplot(grid, grid, i + 1)
            plt.imshow(images[i].numpy().astype('uint8'))
            plt.title(self.class_names[self.labels[i]])
            plt.axis('off')
            
            
    def transferDataFromGoogleDrive(currentDirectory,destinationDirectory,
                                localDirectory):
        
        t0 = time() 
        copyfile(currentDirectory,destinationDirectory)
        print(f"File extraction completed in {(time() - t0)} seconds")
        os.chdir(localDirectory)
        os.system(f"tar -xf {destinationDirectory}")
        os.chdir(self.config['pathDirectories']['root'])
        os.remove(destinationDirectory)
        print(f"File extraction completed in {(time() - t0)} seconds")
        
                
if __name__ == '__main__' :
  preprocessing = Preprocessing(labels = 'inferred', label_mode = 'int',
                                class_names= ['0','90','180','270'],color_mode = 'rgb',
                                batch_size =  256, image_size = (224, 224), # BiT image size > 96 x 96 px # VGG16 (224, 224) # Xception (299, 299)
                                seed = 1234, validation_split = 0.2,
                                interpolation= 'bicubic', num_classes = 4 ) 
