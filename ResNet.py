import torch
import torch.nn as nn
import timeit


# Get the accuracy
def accuracy(output, label):
    _, pred = torch.max(output, dim=1)
    return torch.tensor(torch.sum(pred == label).item() / len(pred))


# Base function for the ImageClassification. It handles all the steps throughout running the model
# It allows more flexibility for recording data through the training of the model
class ImageClassification(nn.Module):
    # Feed forward
    def training_step(self, batch):
        image, label = batch
        output = self(image)
        loss = torch.nn.functional.cross_entropy(output, label)
        return loss

    # Feed forward during training
    def testing_step(self, batch):
        image, label = batch
        output = self(image)
        loss = torch.nn.functional.cross_entropy(output, label)
        acc = accuracy(output, label)
        return {'loss': loss.detach(), 'acc': acc}

    # Collects several stats necessary to see how the model is performaning during training, namely
    # accuracy and loss
    def testing_end(self, output):
        batch_loss = [x['loss'] for x in output]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_accuracy = [x['acc'] for x in output]
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {'loss': epoch_loss.item(), 'acc': epoch_accuracy.item()}

    # Displays the stats at the end of the epoch
    def epoch_end(self, epoch, result, time, file):
        print(f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.5f}, train_loss: {result['test_loss']:.4f}, "
              f"test_loss: {result['loss']:.4f}, acc: {result['acc']:.4f}, Epoch_time: {time}")
        file.write(f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.5f}, train_loss: {result['test_loss']:.4f}, "
                   f"test_loss: {result['loss']:.4f}, acc: {result['acc']:.4f}, Epoch_time: {time}\n")


# This is an implementation of the Resnet9 neural network. not sure if it's good
class ResNet(ImageClassification):
    def __init__(self, in_channels, num_class):
        super().__init__()

        self.conv1 = create_conv_layer(in_channels, 64)
        self.conv2 = create_conv_layer(64, 128, pool=True)
        self.res1 = nn.Sequential(create_conv_layer(128, 128), create_conv_layer(128, 128))
        self.conv3 = create_conv_layer(128, 256, pool=True)
        self.conv4 = create_conv_layer(256, 512, pool=True)
        self.res2 = nn.Sequential(create_conv_layer(512, 512), create_conv_layer(512, 512, ))
        self.classifier = nn.Sequential(nn.MaxPool2d(4), nn.Flatten(), nn.Linear(512, num_class))

    def forward(self, inputs):
        output = self.conv1(inputs)
        output = self.conv2(output)
        output = self.res1(output) + output
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.res2(output) + output
        output = self.classifier(output)
        return output


# Create a layer of the convolutional layer
def create_conv_layer(in_channels, out_channels, activation='relu', pool=False, normalize=True, init=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]

    if init and activation == 'relu':
        torch.nn.init.kaiming_uniform_(layers[0].weight, nonlinearity='relu')
    elif init and activation == 'leaky':
        torch.nn.init.kaiming_uniform_(layers[0].weight)

    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))

    if activation == 'relu':
        layers.append(nn.ReLU(inplace=True))
    elif activation == 'leaky':
        layers.append(nn.LeakyReLU(inplace=True))

    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


# Test function
@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    output = [model.testing_step(batch) for batch in loader]
    return model.testing_end(output)


# Returns the learning rate
def get_learning_rate(optimizer):
    for param in optimizer.param_groups:
        return param['lr']


# Main training function
def cycle(epochs, max_lr, model, trn_dataloader, tst_dataloader, weight_decay=0, max_acc=65, grad_clip=None,
          opt_func=torch.optim.SGD, file=None):
    torch.cuda.empty_cache()
    history = []

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
    #                                                 steps_per_epoch=len(trn_dataloader))

    # file.write(f'\nmax_lr: {max_lr}, weight_decay: {weight_decay}, grad_clip: {grad_clip}, optimizer: {opt_func}\n')
    # file.write('----------------------------------------------------------------------------------------------\n\n')
    for epoch in range(1, epochs + 1):
        start = timeit.default_timer()
        model.train()
        trn_loss = []
        lr_list = []
        for batch in trn_dataloader:
            loss = model.training_step(batch)
            trn_loss.append(loss)
            loss.backward()

            # Gradient clipping
            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)

            optimizer.step()
            optimizer.zero_grad()

            lr_list.append(get_learning_rate(optimizer))
            # scheduler.step()

        result = evaluate(model, tst_dataloader)
        result['test_loss'] = torch.stack(trn_loss).mean().item()
        result['lrs'] = lr_list
        end = timeit.default_timer() - start
        model.epoch_end(epoch, result, end, file)
        history.append(result)
        if result['acc'] >= max_acc:
            break
    return history
