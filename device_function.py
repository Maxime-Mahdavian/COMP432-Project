import torch

# For using a GPU if available
def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')


def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


# Warp a dataloader to move data to device
class DeviceDataloader():
    def __init__(self, dataloader, device):
        self.dataloader = dataloader
        self.device = device

    def __iter__(self):
        for x in self.dataloader:
            yield to_device(x, self.device)

    # Number of batches
    def __len__(self):
        return len(self.dataloader)