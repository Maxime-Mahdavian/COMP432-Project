from tabnanny import verbose
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
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

# CallBack that finishes training if training accuracy is greater than the treshold
class AccuracyTresholdCallBack(tensorflow.keras.callbacks.Callback):
    def __init__(self, threshold):
        super(AccuracyTresholdCallBack, self).__init__()
        self.threshold = threshold

    def on_epoch_end(self, epoch, logs=None): 
        acc = logs["accuracy"]
        if acc >= self.threshold:
            self.model.stop_training = True

# CallBack that updates the learning rate and stores the loss after each batch
class MultiplicativeLearningRate(tensorflow.keras.callbacks.Callback):
    
    def __init__(self, factor):
        self.factor = factor
        self.losses = []
        self.learning_rates = []
        
    def on_batch_end(self, batch, logs):
        loss_logs = logs["loss"]
        self.learning_rates.append(tensorflow.keras.backend.get_value(self.model.optimizer.lr))
        self.losses.append(loss_logs)
        tensorflow.keras.backend.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)

# Method that finds the optimal learning rate for the model
def find_lr(model, X, validation_data, batch_size, min_lr = 10**-15, max_lr=10**-2):
    
    # Compute learning rate multiplicative factor
    num_iter = len(X)
    lr_factor = np.exp(np.log(max_lr / min_lr) / num_iter)
    
    # Train for 1 epoch, starting with minimum learning rate and increase it
    tensorflow.keras.backend.set_value(model.optimizer.lr, min_lr)
    lr_callback = MultiplicativeLearningRate(lr_factor)
    model.fit(X, epochs=1, validation_data=validation_data, batch_size=batch_size, callbacks=[lr_callback])
    
    # Plot loss vs log-scaled learning rate
    plt.plot(lr_callback.learning_rates, lr_callback.losses)
    plt.xscale("log") 
    plt.xlabel("Learning Rate (log-scale)")
    plt.ylabel("Training Loss")
    plt.title("Optimal learning rate is slightly below minimum")
    plt.show()

# Method that plots the metrics after training is complete
def plot_metrics(training_accuracy, validation_accuracy, 
    training_precision, validation_precision, 
    training_loss, validation_loss):

    fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=(20,5))
    fig.suptitle("METRICS")

    ax1.plot(range(1, len(training_accuracy) + 1), training_accuracy)
    ax1.plot(range(1, len(validation_accuracy) + 1), validation_accuracy)
    ax1.set_title('Accuracy')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('accuracy')
    ax1.legend(['training', 'validation'])


    ax2.plot(range(1, len(training_loss) + 1), training_loss)
    ax2.plot(range(1, len(validation_loss) + 1), validation_loss)
    ax2.set_title('Loss')
    ax2.set_xlabel('epochs')
    ax2.set_ylabel('loss')
    ax2.legend(['training', 'validation'])
    
    ax3.plot(range(1, len(training_precision) + 1), training_precision)
    ax3.plot(range(1, len(validation_precision) + 1), validation_precision)
    ax3.set_title('Precision')
    ax3.set_xlabel('epochs')
    ax3.set_ylabel('precision')
    ax3.legend(['training', 'validation'])

    plt.show()

# Data augmentation to train the model with more data, so it generalizes better
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        validation_split = 0.2,                         
        rotation_range=5,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
        )

valid_datagen = ImageDataGenerator(
    rescale = 1./255,
    validation_split = 0.2
    )

test_datagen  = ImageDataGenerator(
    rescale = 1./255
    )

# Get the training and test data and initialize train/validation/test sets
train_dataset = train_datagen.flow_from_directory(directory="dataset/train", target_size=(48, 48), class_mode="categorical", subset="training", batch_size=64)

valid_dataset = train_datagen.flow_from_directory(directory="dataset/val", target_size=(48, 48), class_mode="categorical", subset="validation", batch_size=64)

test_dataset = train_datagen.flow_from_directory(directory="dataset/test", target_size=(48, 48), class_mode="categorical", batch_size=64)

# VGG16 is a simple and widely used Convolutional Neural Network Architecture used for ImageNet,
# a large visual database project used in visual object recognition software research
# base_model = VGG16(input_shape = (48, 48, 3), include_top = False, weights = None)
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
# model.add(Activation("relu"))
model.add(Activation(tensorflow.keras.layers.LeakyReLU(alpha=0.03)))
# Add Dropout, which is a regularization method that efficiently approximates training a large number of neural networks and avoids co-adaptations
# model.add(Dropout(0.5))
# Add a Dense layer at the top of the Convolution layer to classify the images
model.add(Dense(32, kernel_initializer="he_uniform"))
# Add Batch normalization, which is similar to input normalization, but it is computed at minibatch level in the internal layers
model.add(BatchNormalization())
# Add relu(Rectified Linear Unit) activation to each layers so that all the negative values are not passed to the next layer
# model.add(Activation("relu"))
model.add(Activation(tensorflow.keras.layers.LeakyReLU(alpha=0.03)))
# Add Dropout, which is a regularization method that efficiently approximates training a large number of neural networks and avoids co-adaptations
# model.add(Dropout(0.5))
# Add a Dense layer at the top of the Convolution layer to classify the images
model.add(Dense(32, kernel_initializer="he_uniform"))
# Add Batch normalization, which is similar to input normalization, but it is computed at minibatch level in the internal layers
model.add(BatchNormalization())
# Add relu(Rectified Linear Unit) activation to each layers so that all the negative values are not passed to the next layer
# model.add(Activation("relu"))
model.add(Activation(tensorflow.keras.layers.LeakyReLU(alpha=0.03)))
# 7 unit Dense layer since we have 7 classes to predict 
model.add(Dense(7, activation="softmax"))

model.summary()

metrics = [
    BinaryAccuracy(name="accuracy"),
    Precision(name="precision"),
    Recall(name="recall")
]

# Use Adam optimizer to reach the global minima while training the model
# If it is stuck in local minima while training then the Adam optimiser will help to get out of local minima and reach global minima
opt = tensorflow.keras.optimizers.SGD(learning_rate=0.001)
# opt = tensorflow.keras.optimizers.Adam()
model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=metrics)

accuracy_callback = AccuracyTresholdCallBack(0.65)

# find_lr(model, train_dataset, valid_dataset, batch_size=64)

train_history = model.fit(train_dataset, validation_data=valid_dataset, epochs=5, verbose=1, callbacks=[accuracy_callback])
test_score = model.evaluate(test_dataset, verbose=1)

# plot_metrics(train_history.history['accuracy'], train_history.history['val_accuracy'], train_history.history['precision'], train_history.history['val_precision'], train_history.history['loss'], train_history.history['val_loss'])
print("Test loss: " + str(test_score[0]))
print("Test accuracy: " + str(test_score[1]))