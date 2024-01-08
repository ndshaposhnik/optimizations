import torchvision
from torch.utils.data import DataLoader, DistributedSampler
import torch.utils.data as data_utils


def setup_data_loaders(rank, world_size, transposition, batch_size=64, download=False, use_cuda=False):
    root = './data'
    
    train_dataset = torchvision.datasets.MNIST(
        root=root, train=True, transform=torchvision.transforms.ToTensor(), download=download
    )
    test_dataset = torchvision.datasets.MNIST(
        root=root, train=False, transform=torchvision.transforms.ToTensor(), download=download
    )

    train_dataset_sub = data_utils.Subset(train_dataset, transposition)

    kwargs = {'num_workers': 4, 'pin_memory': use_cuda}

    sampler_train = DistributedSampler(train_dataset_sub,
                                 num_replicas=world_size,
                                 rank=rank,
                                 shuffle=True,
                                 seed=42)
    
    sampler_test = DistributedSampler(test_dataset,
                                 num_replicas=world_size,
                                 rank=rank,
                                 shuffle=True,
                                 seed=42)
    
    train_loader = DataLoader(dataset=train_dataset_sub,
                              batch_size=batch_size,
                              shuffle=False, 
                              sampler=sampler_train,
                              **kwargs)
    

    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, sampler=sampler_test, shuffle=False, **kwargs
    )
    
    return train_loader, test_loader