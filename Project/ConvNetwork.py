import torch
import torch.nn as nn
import ImageClassification


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
def conv_block(in_channels, out_channels, pool=False):
    layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
              nn.BatchNorm2d(out_channels),
              nn.ReLU(inplace=True)]
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


def training_step(epochs, max_lr, model, trn_dataloader, tst_dataloader, weight_decay=0, grad_clip=None,
                  opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                    steps_per_epoch=len(trn_dataloader))

    for epoch in range(epochs):
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
        model.epoch_end(epoch, result)
        history.append(result)
    return history
