import torchvision
from torch.utils.data import DataLoader
import torch.utils.data as data_utils

def base_setup_data_loaders(transposition, batch_size=64, download=False, use_cuda=False):
    root = './data'
    
    train_dataset = torchvision.datasets.MNIST(
        root=root, train=True, transform=torchvision.transforms.ToTensor(), download=download
    )
    test_dataset = torchvision.datasets.MNIST(
        root=root, train=False, transform=torchvision.transforms.ToTensor(), download=download
    )

    train_dataset_sub = data_utils.Subset(train_dataset, transposition)

    kwargs = {'num_workers': 0, 'pin_memory': use_cuda}
    
    train_loader = DataLoader(dataset=train_dataset_sub,
                              batch_size=batch_size,
                              shuffle=False,
                              **kwargs)
    

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False, **kwargs
    )
    
    return train_loader, test_loader
