import torch
import torch.nn as nn
import timeit


# Get the accuracy
def accuracy(output, label):
    _, pred = torch.max(output, dim=1)
    return torch.tensor(torch.sum(pred == label).item() / len(pred))


# Base function for the ImageClassification. It handles all the steps throughout running the model
# I am willing to change how it's implemented, but that's how I found how to do it, so I just went with it
class ImageClassification(nn.Module):
    def training_step(self, batch):
        image, label = batch
        output = self(image)
        loss = torch.nn.functional.cross_entropy(output, label)
        return loss

    def testing_step(self, batch):
        image, label = batch
        output = self(image)
        loss = torch.nn.functional.cross_entropy(output, label)
        acc = accuracy(output, label)
        return {'loss': loss.detach(), 'acc': acc}

    def testing_end(self, output):
        batch_loss = [x['loss'] for x in output]
        epoch_loss = torch.stack(batch_loss).mean()
        batch_accuracy = [x['acc'] for x in output]
        epoch_accuracy = torch.stack(batch_accuracy).mean()
        return {'loss': epoch_loss.item(), 'acc': epoch_accuracy.item()}

    def epoch_end(self, epoch, result, time, file):
        print(f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.5f}, train_loss: {result['test_loss']:.4f}, "
              f"test_loss: {result['loss']:.4f}, acc: {result['acc']:.4f}, Epoch_time: {time}")
        file.write(f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.5f}, train_loss: {result['test_loss']:.4f}, "
                   f"test_loss: {result['loss']:.4f}, acc: {result['acc']:.4f}, Epoch_time: {time}\n")


# This is an implementation of the Resnet9 neural network. not sure if it's good
class ConvNetwork(ImageClassification):
    def __init__(self, in_channels, num_class):
        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))
        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))
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
def conv_block(in_channels, out_channels, pool=False, normalize=True):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)]

    if normalize:
        layers.append(nn.BatchNorm2d(out_channels))

    layers.append(nn.ReLU(inplace=True))

    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    output = [model.testing_step(batch) for batch in loader]
    return model.testing_end(output)


def get_learning_rate(optimizer):
    for param in optimizer.param_groups:
        return param['lr']


def cycle(epochs, max_lr, model, trn_dataloader, tst_dataloader, weight_decay=0, grad_clip=None,
          opt_func=torch.optim.SGD, file=None):
    torch.cuda.empty_cache()
    history = []

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                    steps_per_epoch=len(trn_dataloader))

    file.write(f'max_lr: {max_lr}, weight_decay: {weight_decay}, grad_clip: {grad_clip}, optimizer: {opt_func}\n')
    file.write('----------------------------------------------------------------------------------------------\n\n')
    for epoch in range(epochs):
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
            scheduler.step()

        result = evaluate(model, tst_dataloader)
        result['test_loss'] = torch.stack(trn_loss).mean().item()
        result['lrs'] = lr_list
        end = timeit.default_timer() - start
        model.epoch_end(epoch, result, end, file)
        history.append(result)
        if result['acc'] >= 0.65:
            break
    return history
