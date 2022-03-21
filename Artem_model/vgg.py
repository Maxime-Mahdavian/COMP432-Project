import numpy as np
import sklearn
import matplotlib.pyplot as plt
import tensorflow
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten, Dropout,BatchNormalization ,Activation
from keras.models import Model, Sequential
from keras.applications.nasnet import NASNetLarge
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping
from keras.optimizer_v1 import Adam
from keras.metrics import BinaryAccuracy
from keras.metrics import Precision
from keras.metrics import Recall

# Data augmentation to train the model with more data, so it generalizes better
train_datagen = ImageDataGenerator(
        # rescale = 1./255,
        validation_split = 0.2,                         
        # rotation_range=5,
        # width_shift_range=0.2,
        # height_shift_range=0.2,
        # shear_range=0.2,
        # horizontal_flip=True,
        # vertical_flip=True,
        # fill_mode='nearest'
        )

valid_datagen = ImageDataGenerator(
    # rescale = 1./255,
    validation_split = 0.2
    )

test_datagen  = ImageDataGenerator(
    # rescale = 1./255
    )

# Get the training and test data and initialize train/validation/test sets
train_dataset = train_datagen.flow_from_directory(directory="face_recognition/train", target_size=(48, 48), class_mode="categorical", subset="training", batch_size=64)

valid_dataset = train_datagen.flow_from_directory(directory="face_recognition/train", target_size=(48, 48), class_mode="categorical", subset="validation", batch_size=64)

test_dataset = train_datagen.flow_from_directory(directory="face_recognition/test", target_size=(48, 48), class_mode="categorical", batch_size=64)

# VGG16 is a simple and widely used Convolutional Neural Network Architecture used for ImageNet,
# a large visual database project used in visual object recognition software research
base_model = VGG16(input_shape = (48, 48, 3), include_top = False, weights = "imagenet")

# We want to have a deep neural network, but we do not want to spend much time training it
# That is why, we freeze the weights of the layers, and use a pretrained model with already useful weights
for layer in base_model.layers[:-4]:
    layer.trainable=False


model = Sequential()
model.add(base_model)
# Add Dropout, which is a regularization method that efficiently approximates training a large number of neural networks and avoids co-adaptations
# model.add(Dropout(0.5))
# Add a Flatten layer to squash the 3 dimensions of an image to a single dimension
model.add(Flatten())
# Batch normalization is similar to input normalization, but it is computed at minibatch level in the internal layers
model.add(BatchNormalization())
# Add a Dense layer at the top of the Convolution layer to classify the images
model.add(Dense(32, kernel_initializer="he_uniform"))
# Add Batch normalization, which is similar to input normalization, but it is computed at minibatch level in the internal layers
model.add(BatchNormalization())
# Add relu(Rectified Linear Unit) activation to each layers so that all the negative values are not passed to the next layer
model.add(Activation("relu"))
# Add Dropout, which is a regularization method that efficiently approximates training a large number of neural networks and avoids co-adaptations
# model.add(Dropout(0.5))
# Add a Dense layer at the top of the Convolution layer to classify the images
model.add(Dense(32, kernel_initializer="he_uniform"))
# Add Batch normalization, which is similar to input normalization, but it is computed at minibatch level in the internal layers
model.add(BatchNormalization())
# Add relu(Rectified Linear Unit) activation to each layers so that all the negative values are not passed to the next layer
model.add(Activation("relu"))
# Add Dropout, which is a regularization method that efficiently approximates training a large number of neural networks and avoids co-adaptations
# model.add(Dropout(0.5))
# Add a Dense layer at the top of the Convolution layer to classify the images
model.add(Dense(32, kernel_initializer="he_uniform"))
# Add Batch normalization, which is similar to input normalization, but it is computed at minibatch level in the internal layers
model.add(BatchNormalization())
# Add relu(Rectified Linear Unit) activation to each layers so that all the negative values are not passed to the next layer
model.add(Activation("relu"))
# 7 unit Dense layer since we have 7 classes to predict 
model.add(Dense(7, activation="softmax"))

model.summary()

metrics = [
    BinaryAccuracy(name="accuracy"),
    Precision(name="precision"),
    Recall(name="recall")
]

# Use Adam optimizer to reach the global minima whilce training the model
# If it is stuck in local minima while training then the Adam optimiser will help to get out of local minima and reach global minima
# 
model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=metrics)

model.fit(train_dataset, validation_data=valid_dataset, epochs=5, verbose=1)
