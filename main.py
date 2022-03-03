import os
import numpy as np
import matplotlib
import torch
import torchvision
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
import torchvision.transforms as tt
from torch.utils.data import random_split
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import timeit
import datetime
import device_function
import ConvNetwork as convnet

data_dir = 'dataset'

# Some hyperparameter and objective function
batch_size = 200
epoch = 50
max_lr = 0.001
grad_clip = 0.1
weight_decay = 1e-4
obj_func = torch.optim.Adam

file = open('result.txt', 'a')

# Show part of a batch
def show_batch(dataloader):
    for images, label in dataloader:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        plt.show()
        break


def plot_accuracy(history):
    acc = [x['acc'] for x in history]
    plt.plot(acc, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('something')
    plt.show()


# print(os.listdir(data_dir))
# classes = os.listdir(data_dir + '/train')
# print(classes)

# Normalizes the data by setting the mean to 0 and variance by 1.
# Also data augmentation, pads the image by 4 pixels, then take a random crop of 32x32 and then
# there is a 50% chance of the image being flipped horizontally
# This happens at every epoch, so the model sees slightly different versions of the images every time
train_tfms = tt.Compose([tt.RandomCrop(48, padding=4, padding_mode='reflect'),
                         tt.RandomHorizontalFlip(),
                         tt.ToTensor()])

test_tfms = tt.Compose([tt.ToTensor()])

# Datasets
train_dataset = ImageFolder(data_dir + '/train', train_tfms)
test_dataset = ImageFolder(data_dir + '/test', test_tfms)

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size * 3, num_workers=3, pin_memory=True)

# show_batch(train_dataloader)

# Get the device that will be used to train the model
# It will be cuda if your GPU is available or cpu
device = device_function.get_device()
print(device)

# Wrap the dataloader to put it to the device
train_dataloader = device_function.DeviceDataloader(train_dataloader, device)
test_dataloader = device_function.DeviceDataloader(test_dataloader, device)

# Create the model ConvNetwork and put it on the device
model = device_function.to_device(convnet.ConvNetwork(3, 7), device)


# History is just to get the history of the loss and accuracy, mostly to see how the model
# evolves
history = [convnet.evaluate(model, test_dataloader)]
#
#
start = timeit.default_timer()
# # Training step is the main training function
history += convnet.cycle(epoch, max_lr, model, train_dataloader, test_dataloader,
                         grad_clip=grad_clip, weight_decay=weight_decay, opt_func=obj_func, file=file)

duration = timeit.default_timer() - start
plot_accuracy(history)
print(datetime.timedelta(seconds=duration))

