import numpy as np
import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import pickle
import argparse

from local_datasets.datasets import get_dataloaders
from DNN_models import *

def validation(model, criterion, test_loader):
    model.eval()
    loss_record = 0.0
    acc_record = 0
    # Taking the average of all metrics:
    for j, (X_batch, y_batch) in enumerate(test_loader):
        y_pred = model(X_batch.cuda())
        loss_ele = criterion(y_pred, y_batch.cuda()).cpu().item()
        acc_ele = (y_batch == y_pred.cpu().max(1)[1]).float().mean().cpu().item()
        loss_record += loss_ele
        acc_record += acc_ele
    j += 1
    loss_record /= j
    acc_record /= j
    return loss_record, acc_record

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', dest='directory', type=str, help='directory')
    parser.add_argument('--p', dest='p', type=float, help='dropout probability')
    parser.add_argument('--seed', dest='seed', type=int, help='random seed')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dir = args.directory
    EXP_PATH = 'checkpoints/vgg11_fashionmnist/' + dir + "/"

    # not very clean way for sharing and saving setup of the experiment
    exec(open(EXP_PATH + "params.py").read())

    epochs = 150

    # Inspection
    inspect_step = 1 #0

    # Optimizer
    lr = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    criterion = torch.nn.CrossEntropyLoss()

    # Get original loss:
    print("-------Initialization----------")
    loss, acc = validation(model, criterion, data_loaders['test'])
    print("Loss", round(loss, 4))

    train_history = {}
    test_history = {}

    # Training:
    for i in range(epochs):
        model.train()
        loss_record = 0.0
        acc_record = 0
        for k, (X_batch, y_batch) in enumerate(data_loaders['train']):
            optimizer.zero_grad()
            y_pred = model(X_batch.cuda())
            loss = criterion(y_pred, y_batch.cuda())
            acc_ele = (y_batch == y_pred.cpu().max(1)[1]).float().mean().cpu().item()
            loss.backward()
            optimizer.step()
            loss_record += loss.cpu().item()
            acc_record += acc_ele
        scheduler.step()

        k += 1
        loss_record /= k
        acc_record /= k
        print("Epoch", i+1)
        print("Training loss", round(loss_record, 4), "Training accuracy", round(acc_record, 4))

        if i == 0:
            train_history = {'loss': [loss_record], 'acc': [acc_record]}
            train_history["epoch"] = [1]
        else:
            train_history['loss'].append(loss_record)
            train_history['acc'].append(acc_record)
            train_history["epoch"].append(i+1)

        if (i+1)%10 == 0:
            torch.save(model.state_dict(), EXP_PATH+"chkp_"+str(i+1))

        if (i+1) % inspect_step == 0 or i==0:
            val_loss, val_acc = validation(model, criterion, data_loaders['test'])
            print("#####Test loss", round(val_loss, 4), "Test accuracy", round(val_acc, 4))
            if i == 0:
                test_history = {'loss': [val_loss], 'acc': [val_acc]}
                test_history["epoch"] = [1]
            else:
                test_history["epoch"].append(i+1)
                test_history['loss'].append(val_loss)
                test_history['acc'].append(val_acc)

    pickle.dump(test_history, open(EXP_PATH + "test_history.pkl", "wb"))
    pickle.dump(train_history, open(EXP_PATH + "train_history.pkl", "wb"))
