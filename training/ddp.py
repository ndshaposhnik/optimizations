import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torchmetrics import Accuracy
from collections import defaultdict
from tqdm import tqdm
from torch import nn
from model.ToyModel import ToyModel
import numpy as np
import os
import timm
import time
from data.ddp_loader import setup_data_loaders


def train(
    rank,
    model, 
    criterion,
    optimizer,
    train_loader,
    val_loader,
    world_size,
    checkpoint_path,
    history_path,
    num_epochs=2,
    val_frequency=1,
    use_fp16=False,
    scaler=None):
    history = defaultdict(lambda: defaultdict(list))
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        val_loader.sampler.set_epoch(epoch)
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
                for param in model.parameters():
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(X_batch)
                loss = criterion(logits, y_batch.long())
                loss.backward()
                for param in model.parameters():
                    dist.all_reduce(param.grad.data, op=dist.ReduceOp.SUM)
                    param.grad.data /= world_size
                optimizer.step()
            optimizer.zero_grad()
            train_loss += loss.detach()*len(X_batch)
            num += len(X_batch)
            duration = torch.Tensor([time.time() - start_time])
            dist.all_reduce(duration, op=dist.ReduceOp.MAX)
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
                acc = Accuracy(task="multiclass", num_classes=10)(logits.max(1)[1], y_batch)*len(X_batch)/world_size
                dist.all_reduce(acc, op=dist.ReduceOp.SUM) 
                val_acc += acc
                num += len(X_batch)
            val_acc /= num
            history['acc']['val'].append(val_acc)

    if rank == 0:
        torch.save(model.state_dict(), checkpoint_path)
        os.mkdir(history_path)
        np.save(history_path + "/train_loss", np.array(history['loss']['train']))
        np.save(history_path + "/time_epoch", np.array(history['time']['epoch']))
        np.save(history_path + "/time_batch", np.array(history['time']['batch']))
        np.save(history_path + "/val_acc", np.array(history['acc']['val']))
    


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)
    torch.manual_seed(42)

def cleanup():
    dist.destroy_process_group()

def ddp_train(rank, world_size, num_epochs, batch_size, checkpoint_path, history_path, transposition, use_mobilenetv3, use_fp16, val_frequency):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)

    train_loader, val_loader = setup_data_loaders(rank, world_size, transposition, batch_size)
    if not use_mobilenetv3:
        model = ToyModel(28*28, 10)
    else:
        model = timm.create_model('mobilenetv3_small_100', in_chans=1, num_classes=10,  pretrained=False) 
    ddp_model = DDP(model)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    scaler = None
    if use_fp16:
        scaler = torch.cuda.amp.GradScaler()

    train(rank, model, 
        loss,
        optimizer, 
        train_loader,
        val_loader,
        world_size,
        checkpoint_path,
        history_path,
        num_epochs,
        val_frequency,
        use_fp16,
        scaler)

def run(ddp_train, world_size, num_epochs, batch_size, checkpoint_path, history_path, transposition, use_mobilenetv3=False, use_fp16=False, val_frequency=1):
    mp.spawn(ddp_train,
             args=(world_size, num_epochs, batch_size, checkpoint_path, history_path, transposition, use_mobilenetv3, use_fp16, val_frequency),
             nprocs=world_size,
             join=True)