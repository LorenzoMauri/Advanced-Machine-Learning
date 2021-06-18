# Image orientation detection

## Overview
Development of a convolutional neural network for the image canonical orientation detection task.
* Classify if an image is correctly oriented (0°) or not (affected by 90°, 180° or 270° orientation)

## Execution
The code has been developed and executed on Google Colaboratory.

To reproduce the same results described in the report of this project, it is necessary to run in the following order:

1. `generator.ipynb` : implements the **dataset generation step** described in the project report.
     * it downloads, processes and saves datasets to Google Drive
     
    As far as the SUN397 dataset is concerned, it was generated locally due to memory limits imposed by Google Colaboratory.

2. `trainer.ipynb` : implements the **training step** described in the project report.
     * it migrates and loads datasets from Google Drive to Google Colaboratory
     * it builds, trains and saves to Google Drive the final model

## Authors

* Vasco Coelho - 807304 - v.coelho@campus.unimib.it
* Lorenzo Mauri - 807306 - l.mauri28@campus.unimib.it
