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
import ResNet as convnet
import argparse
from split_folder import ratio
from orion.client import report_objective

# ratio('dataset2/train', output='output', seed=40093125, ratio=(0.8, 0.2))

data_dir = 'dataset'
torch.manual_seed(0)

parser = argparse.ArgumentParser()
parser.add_argument("-lr", help='learning rate')
parser.add_argument("-batch_size", help='batch size')
args = parser.parse_args()

# Some hyperparameter and objective function
batch_size = int(args.batch_size)
epoch = 50
max_lr = float(args.lr)
grad_clip = 0.1
weight_decay = 1e-4
max_acc = 0.65
obj_func = torch.optim.Adam

file = open('leaky_relu.txt', 'a')


# Show part of a batch
def show_batch(dataloader):
    for images, label in dataloader:
        fig, ax = plt.subplots(figsize=(12, 12))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(images[:64], nrow=8).permute(1, 2, 0))
        plt.show()
        break


def plot_graph(history, title):
    plt.subplot(1, 2, 1)
    acc = [x['acc'] for x in history]
    plt.plot(acc, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')

    plt.subplot(1, 2, 2)
    test_loss = [x['loss'] for x in history]
    train_loss = [x.get('test_loss') for x in history]
    plt.plot(test_loss, '-x')
    plt.plot(train_loss, '-o')
    plt.xlabel('epoch')
    plt.ylabel('Loss')
    plt.title('Training and test loss')
    plt.legend(['test loss', 'training loss'])

    plt.savefig(title)
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
val_dataset = ImageFolder(data_dir + '/val', test_tfms)

# DataLoaders
train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=3, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size * 3, num_workers=3, pin_memory=True)
validation_dataloader = DataLoader(val_dataset, batch_size * 3, num_workers=3, pin_memory=True)

# show_batch(test_dataloader)

# Get the device that will be used to train the model
# It will be cuda if your GPU is available or cpu
device = device_function.get_device()
print(device)

# Wrap the dataloader to put it to the device
train_dataloader = device_function.DeviceDataloader(train_dataloader, device)
test_dataloader = device_function.DeviceDataloader(test_dataloader, device)
validation_dataloader = device_function.DeviceDataloader(validation_dataloader, device)
#
for i in range(3):
    filename = "leaky_relu" + str(i) + ".png"

    # Create the model ConvNetwork and put it on the device
    model = device_function.to_device(convnet.ResNet(3, 7), device)

    # History is just to get the history of the loss and accuracy, mostly to see how the model
    # evolves
    history = [convnet.evaluate(model, validation_dataloader)]
    print(f"Epoch [{0}], "
          f"test_loss: {history[0]['loss']:.4f}, acc: {history[0]['acc']:.4f}, Epoch_time: {0}")
    file.write(f"Epoch [{0}], "
               f"test_loss: {history[0]['loss']:.4f}, acc: {history[0]['acc']:.4f}, Epoch_time: {0}\n")
    #
    #
    start = timeit.default_timer()
    # Training step is the main training function
    history += convnet.cycle(epoch, max_lr, model, train_dataloader, validation_dataloader, max_acc=max_acc,
                             grad_clip=grad_clip, weight_decay=weight_decay, opt_func=obj_func, file=file)

    # Orion helper function
    # report_objective(history[-1]['loss'])

    duration = timeit.default_timer() - start
    plot_graph(history, filename)
    print(datetime.timedelta(seconds=duration))
    file.write(str(datetime.timedelta(seconds=duration)) + "\n")
