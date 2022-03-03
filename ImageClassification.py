import torch
import torch.nn as nn

def accuracy(output, label):
    _, pred = torch.max(output, dim=1)
    return torch.tensor(torch.sum(pred == label).item() / len(pred))

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

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch}], last_lr: {result['lrs'][-1]:.5f}, train_loss: {result['test_loss']:.4f}, "
              f"val_loss: {result['loss']:.4f}, acc: {result['acc']:.4f}")

