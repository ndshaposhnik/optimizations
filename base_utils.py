import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, DistributedSampler
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import Accuracy
from collections import defaultdict
import torch.utils.data as data_utils
from tqdm import tqdm
from torch import nn
from modules import ToyModel
import numpy as np
import os
import timm
import time

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

def train_base(
    model, 
    criterion,
    optimizer,
    train_loader,
    val_loader,
    checkpoint_path,
    history_path,
    num_epochs=2,
    val_frequency=1,
    use_fp16=False,
    scaler=None):
    history = defaultdict(lambda: defaultdict(list))
    for epoch in range(num_epochs):
        train_loss = 0
        val_acc = 0
        model.train(True) 
        num = 0
        train_time = 0
        batch_time = []
        for X_batch, y_batch in tqdm(train_loader):
            start_time = time.time()
            if use_fp16:
                with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
                    logits = model(X_batch)
                    loss = criterion(logits, y_batch.long())
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X_batch)
                loss = criterion(logits, y_batch.long())
                loss.backward()
                optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.detach()*len(X_batch)
            num += len(X_batch)
            duration = torch.Tensor([time.time() - start_time])
            batch_time.append(duration)
            train_time += duration
        train_loss /= num
        history['loss']['train'].append(train_loss)
        history['time']['epoch'].append(train_time)
        history['time']['batch'].append(batch_time)
            
        if epoch%val_frequency == 0:
            model.train(False) 
            num = 0
            for X_batch, y_batch in val_loader:
                logits = model(X_batch)
                acc = Accuracy(task="multiclass", num_classes=10)(logits.max(1)[1], y_batch)*len(X_batch)
                val_acc += acc
                num += len(X_batch)
            val_acc /= num
            history['acc']['val'].append(val_acc)

    torch.save(model.state_dict(), checkpoint_path)
    os.mkdir(history_path)
    np.save(history_path + "/train_loss", np.array(history['loss']['train']))
    np.save(history_path + "/time_epoch", np.array(history['time']['epoch']))
    np.save(history_path + "/time_batch", np.array(history['time']['batch']))
    np.save(history_path + "/val_acc", np.array(history['acc']['val']))
    

def run_base(num_epochs, batch_size, checkpoint_path, history_path, transposition, use_mobilenetv3=False, use_fp16=False, val_frequency=1):
    train_loader, val_loader = base_setup_data_loaders(transposition, batch_size)
    if not use_mobilenetv3:
        model = ToyModel(28*28, 10)
    else:
        model = timm.create_model('mobilenetv3_small_100', in_chans=1, num_classes=10,  pretrained=False) 
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = None
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    train_base(
        model, 
        loss,
        optimizer,
        train_loader,
        val_loader,
        checkpoint_path,
        history_path,
        num_epochs,
        val_frequency,
        use_fp16,
        scaler)