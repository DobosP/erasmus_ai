#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Convolutional Neural Network
 
# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
 
# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html
 
# Installing Keras
# pip install --upgrade keras


# In[2]:


# Part 1 - Building the Models
 
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras.layers import Flatten
from keras.layers import Input
from keras.models import Model

# Multilayer Perceptron
def create_mlp(dim, regress=False):
    # define our MLP network
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))

    # check to see if the regression node should be added
    if regress:
        model.add(Dense(1, activation="linear"))

    # return our model
    return model

# Regression-based CNN
def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    # initialize the input shape and channel dimension, assuming
    # TensorFlow/channels-last ordering
    inputShape = (height, width, depth)
    chanDim = -1

    # define the model input
    inputs = Input(shape=inputShape)

    # loop over the number of filters
    for (i, f) in enumerate(filters):
        # if this is the first CONV layer then set the input
        # appropriately
        if i == 0:
            x = inputs

        # Step 1 - Convolution
        x = Conv2D(f, (3, 3), padding="same")(x)
        
        # Step 2 - RELU
        x = Activation("relu")(x)
        
        # Step 3 - BN
        x = BatchNormalization(axis=chanDim)(x)
        
        # Step 4 - Pooling
        x = MaxPooling2D(pool_size=(2, 2))(x)

    # Step 5 - Flattening
    x = Flatten()(x)
    
    # Step 6 - FC layer
    x = Dense(16)(x)
    
    # Step 7 - RELU
    x = Activation("relu")(x)
    
    # Step 8 - BN
    x = BatchNormalization(axis=chanDim)(x)
    
    # Step 9 - DROPOUT
    x = Dropout(0.5)(x)

    # Step 10 - apply another FC layer, this one to match the number of nodes
    # coming out of the MLP
    x = Dense(4)(x)
    
    # Step 11 - RELU
    x = Activation("relu")(x)

    # check to see if the regression node should be added
    if regress:
        x = Dense(1, activation="linear")(x)

    # construct the CNN
    model = Model(inputs, x)

    # return the CNN
    return model


# In[3]:


from PIL import Image
import io
import os

IMAGE_SIZE = 64

def resize_image(path):
    image = Image.open(path)
    resized_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    # Uncomment to save to local directory 
    # Get image name
    name=os.path.basename(path)
    resized_image.save('C:/Users/madad/Documents/dataset/images/res1/' + name, "JPEG")

    # new_image size (64,64)
    return np.asarray(resized_image)


def read_image(path):
    image = Image.open(path)
    return np.asarray(image)


# In[4]:


# Load data helpers
import pandas as pd
from random import shuffle
import glob
from PIL import Image
import re

def process_csv(df):
    new_df = pd.DataFrame() 

    for index, row in df.iterrows():
        
        quantity = int(row['PhotoAmt'])
        
        if (not re.match("^[a-zA-Z0-9_]*$", row['PetID'])):
            print(row['PetID'])
            continue
    
        petId = str(row['PetID'])
        
        for i in range(1, quantity+1):
            new_row = row
            new_row['PetID'] = petId + '-' + str(i)
            
            new_df = new_df.append([new_row],ignore_index=True)
            
    return new_df
    
def load_pet_attributes(inputPath):
    # initialize the list of column names in the CSV file and then
    # load it using Pandas
    cols = ["Type", "Name", "Age", "Breed1", "Breed2", "Gender", "Color1", "Color2", "Color3", "MaturitySize", "FurLength", "Vaccinated", "Dewormed", "Sterilized", "Health", "Quantity", "Fee", "State", "RescuerID", "VideoAmt", "Description", "PetID", "PhotoAmt", "AdoptionSpeed"]
    df = pd.read_csv(inputPath, header=0, usecols=['PhotoAmt', 'PetID', 'AdoptionSpeed'], names=cols)
    
    df = process_csv(df)
    
    # return the data frame
    return df

def load_pet_images(df, inputDir):    
    # initialize images array 
    images = []
    outputImage = np.zeros((64, 64, 3), dtype="uint8")
    
    # loop over the csv rows
    for index, row in df.iterrows():
        
        img_path = inputDir + row['PetID'] + '.jpg'
    
        resized = read_image(img_path)
        #TODO
        outputImage = resized
        
        # add the image to the set of images the network will be trained on
        images.append(outputImage)

    # return our set of images
    return np.array(images)


# In[5]:


# import the necessary packages
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import argparse
import locale

# construct the path to the train.csv file that contains information
# on each pet in the dataset and then load the dataset
print("[INFO] loading pet features...")
df = load_pet_attributes('C:/Users/madad/Documents/dataset/train.csv')
print("[INFO] processed features")

# load the pet images and then scale the pixel intensities to the
# range [0, 1]
print("[INFO] loading pet images...")
images = load_pet_images(df, 'C:/Users/madad/Documents/dataset/images/resized/')
images = images / 255.0
print("[INFO] processed images")

# partition the data into training and testing splits using 90% of
# the data for training and the remaining 10% for testing
split = train_test_split(df, images, test_size=0.1, random_state=42)
(trainAttrX, testAttrX, trainImagesX, testImagesX) = split

# the largest pet adoption value is 4 
# use it to scale the pet adoption speed to the range [0, 1] (will lead to better
# training and convergence)
maxAdoption = 4
trainY = trainAttrX["AdoptionSpeed"] / maxAdoption
testY = testAttrX["AdoptionSpeed"] / maxAdoption

# create the Convolutional Neural Network and then compile the model
# using mean absolute percentage error as loss, implying that we
# seek to minimize the absolute percentage difference between our
# adoption speed *predictions* and the *actual adoption speeds*
model = create_cnn(64, 64, 3, regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(trainImagesX, trainY, validation_data=(testImagesX, testY),
          epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting pet adoption speed...")
preds = model.predict(testImagesX)

# compute the difference between the *predicted* adoption speeds and the
# *actual* adoption speeds, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# finally, show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. house price: {}, std house price: {}".format(
    locale.currency(df["AdoptionSpeed"].mean(), grouping=True),
    locale.currency(df["AdoptionSpeed"].std(), grouping=True)))
print("[INFO] mean: {:.2f}%, std: {:.2f}%".format(mean, std))


# In[ ]:


from keras.models import model_from_json

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
classifier.save_weights("model.h5")
print("Saved model to disk")

