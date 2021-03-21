import os 
import json 
import tensorflow as tf 
import keras
from matplotlib import pyplot as plt 

def createFolder(name) :
  if not os.path.exists(name):
    os.mkdir(name)

def readConfigFile(filePathConfig):
  # reading configuration file 
  with open(filePathConfig, "r") as config:
      readObjectConfig = config.read()
      config = json.loads(readObjectConfig)

  # reading all the paths
  return config



def dataLoader(trainDataDirectory, testDataDirectory):
  
  # Train
  train_ds = tf.keras.preprocessing.image_dataset_from_directory(
      trainDataDirectory,
      labels=LABELS,
      label_mode=LABEL_MODE,
      class_names=CLASS_NAMES,
      color_mode=COLOR_MODE,
      batch_size=BATCH_SIZE,
      image_size=IMAGE_SIZE,
      shuffle=True,
      seed=SEED,
      validation_split=VALIDATION_SPLIT,
      subset='training',
      interpolation=INTERPOLATION,
      follow_links=False,
  )

  # Validation
  validation_ds = tf.keras.preprocessing.image_dataset_from_directory(
      trainDataDirectory,
      labels=LABELS,
      label_mode=LABEL_MODE,
      class_names=CLASS_NAMES,
      color_mode=COLOR_MODE,
      batch_size=BATCH_SIZE,
      image_size=IMAGE_SIZE,
      shuffle=True,
      seed=SEED,
      validation_split=VALIDATION_SPLIT,
      subset='validation',
      interpolation=INTERPOLATION,
      follow_links=False,
  )

  # Test
  test_ds = tf.keras.preprocessing.image_dataset_from_directory(
      testDataDirectory,
      labels=LABELS,
      label_mode=LABEL_MODE,
      class_names=CLASS_NAMES,
      color_mode=COLOR_MODE,
      batch_size=BATCH_SIZE,
      image_size=IMAGE_SIZE,
      shuffle=True,
      seed=SEED,
      validation_split=None,
      subset=None,
      interpolation=INTERPOLATION,
      follow_links=False,
  )

  # controls 
  assert isinstance(train_ds, tf.data.Dataset)
  print('Number of train batches: %d' % tf.data.experimental.cardinality(train_ds))
  assert isinstance(validation_ds, tf.data.Dataset)
  print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_ds))
  assert isinstance(test_ds, tf.data.Dataset)
  print('Number of test batches: %d' % tf.data.experimental.cardinality(test_ds))

  return train_ds, validation_ds, test_ds



def visualizeImages(grid=3,figsize = (10,10)) : 
  plt.figure(figsize=figsize)
  for images, labels in train_ds.take(1):
    for i in range(grid*grid):
      ax = plt.subplot(grid, grid, i + 1)
      plt.imshow(images[i].numpy().astype('uint8'))
      plt.title(CLASS_NAMES[labels[i]])
      plt.axis('off')
