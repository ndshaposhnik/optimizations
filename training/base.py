import torch
from torchmetrics import Accuracy
from collections import defaultdict
from tqdm import tqdm
from torch import nn
from model.ToyModel import ToyModel
import numpy as np
import os
import timm
import time
from data.base_loader import base_setup_data_loaders

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
    scaler=None,
):
    history = defaultdict(list)
    history['coords'].append(0)
    for epoch in tqdm(range(num_epochs)):
        train_loss = 0
        val_acc = 0
        model.train(True) 
        num = 0
        coords = 0
        for X_batch, y_batch in train_loader:
            logits = model(X_batch)
            loss = criterion(logits, y_batch.long())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            train_loss += loss.detach() * len(X_batch)
            num += len(X_batch)
            coords += optimizer.last_coordinates_transmitted

        history['loss'].append(train_loss / num)
        history['coords'].append(coords)
            
    history['coords'] = history['coords'][:-1]
    torch.save(model.state_dict(), checkpoint_path)
    np.save(history_path + "/loss", np.array(history['loss']))
    np.save(history_path + "/coords", np.array(history['coords']))
    

def run_base(
    num_epochs, 
    batch_size, 
    checkpoint_path, 
    history_path, 
    transposition, 
    use_mobilenetv3=False, 
    use_fp16=False, 
    val_frequency=1,
    **kwargs,
):
    train_loader, val_loader = base_setup_data_loaders(transposition, batch_size)
    if not use_mobilenetv3:
        model = ToyModel(28*28, 10)
    else:
        model = timm.create_model('mobilenetv3_small_100', in_chans=1, num_classes=10,  pretrained=False) 
    loss = nn.CrossEntropyLoss()

    optimizer = torch.optim.CompressedSGD(model.parameters(), lr=0.001, **kwargs)

    history_path += "/" + optimizer.compressor_name
    for i in range(100): # Find first unused number
        end = ''
        if i > 0:
            end = f'_{i}'
        if not os.path.isdir(history_path + end):
            history_path += end
            break

    os.mkdir(history_path)
    with open(history_path + '/info.txt', 'w') as f:
        f.write(str(kwargs))

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
