# Cotton-Disease-Detection

## Table of Contents
 * [Overview](#overview)
 * [Purpose](#Purpose)
 * [Codes and Resources Used](#CodesandResourcesUsed)
 * [Model preprocessing and building](#Modelpreprocessingandbuilding)
 * [Future Scope](#FutureScope)


## Overview
Cotton Disease Predictor is a FastAPI web application which classifies a cotton plant/leaf image into four categories
1. Fresh cotton leaf
2. Fresh cotton plant
3. Diseased cotton leaf
4. Diseased cotton plant
The code is written in Python and makes use of Tensorflow libraries in developing Image classification web application and deployed in Heroku.

## Purpose
* India is the largest producer of cotton in the world but recently the production of cotton in india reducing gradualy over year because of major cotton diseases  which impact their production.
*  The issue will be resolved  if the farmer gets to know about the plants which are infected and diseased in early stages of their growth.
*  This project will help the farmers to recognize the cotton plants which are Fresh and Diseased by simply uploading the pictures of the cotton plants on the web app.


Link : [https://cotton-disease-dl.herokuapp.com/docs](https://cotton-disease-dl.herokuapp.com/docs)


![Front end](https://imgur.com/aE4S8TD.png)

## Codes and Resources Used
* Python Version: 3.8
* Packages: pandas, numpy, tensorflow, fastapi, opencv-python-headless

* For Web Framework Requirements: pip install -r requirements.txt
* Dataset: https://www.kaggle.com/datasets/janmejaybhoi/cotton-disease-dataset
* FastAPI Productionization:
 https://fastapi.tiangolo.com/tutorial/
 https://www.tutlinks.com/create-and-deploy-fastapi-app-to-heroku/
 
## Model preprocessing and building
* Performed resize and rescale and Data Augmentation
* Built the model architecture using CNN.
* Added multiple max pooling and Conv 2D layers.
* Then flattened the layer and added dense layer with softmax activation function.
* Fit the model with 50 epochs and provided the following results.

![Front end](https://imgur.com/y2v9Eom.png)

## Future Scope

* Use different different Transfer Learning Architecture.
* Build a mobile app.
* Deploy on different platforms : GCP, Azure.


 
 
 
 
 
