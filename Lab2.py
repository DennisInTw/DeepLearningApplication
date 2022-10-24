import os

import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import logging
import sys


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # ==========================
        # TODO 1: build your network
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)

        self.conv5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.conv8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(in_features=(512 * 7 * 7), out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=4096)
        self.fc3 = nn.Linear(in_features=4096, out_features=1000)
        self.fc4 = nn.Linear(in_features=1000, out_features=10)
#        self.fc5 = nn.Linear(in_features=100, out_features=10)
#        self.fc6 = nn.Linear(in_features=64, out_features=10)
#        self.fc7 = nn.Linear(in_features=128, out_features=64)
#        self.fc8 = nn.Linear(in_features=64, out_features=32)
#        self.fc9 = nn.Linear(in_features=32, out_features=16)
#        self.fc10 = nn.Linear(in_features=16, out_features=10)
        # ==========================

    def forward(self, x):
        # (batch_size, 3, 256, 256)

        # ========================
        # TODO 2: forward the data
        # please write down the output size of each layer
        # example:
        # out = self.relu(self.conv1(x))
        # (batch_size, 64, 256, 256)

        out = self.relu(self.conv1(x))
        # (batch_size, 64, 256, 256)
        out = self.relu(self.conv2(out))
        # (batch_size, 64, 256, 256)
        out = self.pool(out)
        # (batch_size, 64, 128, 128)

        out = self.relu(self.conv3(out))
        # (batch_size, 128, 128, 128)
        out = self.relu(self.conv4(out))
        # (batch_size, 128, 128, 128)
        out = self.pool(out)
        # (batch_size, 128, 64, 64)

        out = self.relu(self.conv5(out))
        # (batch_size, 256, 64, 64)
        out = self.relu(self.conv6(out))
        # (batch_size, 256, 64, 64)
        out = self.relu(self.conv7(out))
        # (batch_size, 256, 64, 64)
        out = self.pool(out)
        # (batch_size, 256, 32, 32)

        out = self.relu(self.conv8(out))
        # (batch_size, 512, 32, 32)
        out = self.relu(self.conv9(out))
        # (batch_size, 512, 32, 32)
        out = self.relu(self.conv10(out))
        # (batch_size, 512, 32, 32)
        out = self.pool(out)
        # (batch_size, 512, 16, 16)

        out = self.relu(self.conv11(out))
        # (batch_size, 512, 16, 16)
        out = self.relu(self.conv12(out))
        # (batch_size, 512, 16, 16)
        out = self.relu(self.conv13(out))
        # (batch_size, 512, 16, 16)
        out = self.pool(out)
        # (batch_size, 512, 8, 8)

        out = torch.flatten(out, 1)
        # (batch_size, 512 * 8 * 8)

        out = self.relu(self.fc1(out))
        # (batch_size, 4096)
        out = self.relu(self.fc2(out))
        # (batch_size, 2048)
        out = self.relu(self.fc3(out))
        # (batch_size, 1024)
        out = self.fc4(out)
        # (batch_size, 512)
#        out = self.fc5(out)
        # (batch_size, 256)
#        out = self.fc6(out)
        # (batch_size, 128)
#        out = self.relu(self.fc7(out))
        # (batch_size, 64)
#        out = self.relu(self.fc8(out))
        # (batch_size, 32)
#        out = self.relu(self.fc9(out))
        # (batch_size, 16)
#        out = self.fc10(out)
        # (batch_size, 10)

        # ========================
        return out


def calc_acc(output, target):
    # Get the maximum in each row of output, then get its index
    # output : (batch_size, 10) ==> (batch_size, 1) ==> get index : (batch_size, 1)
    predicted = torch.max(output, 1)[1]
    num_samples = target.size(0)
    num_correct = (predicted == target).sum().item()
#    print((predicted == target))
#    print((predicted == target).sum())
#    print((predicted == target).sum().item())
    return num_correct / num_samples


def training(model, device, train_loader, criterion, optimizer):
    # ===============================
    # TODO 3: switch the model to training mode
    # Actually, we can ignore this setting because when self.training=True by default
    model.train(mode=False)
    # ===============================
    train_acc = 0.0
    train_loss = 0.0

    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
#        logging.info(f'[DBG] training data shape : {data.shape}')

        # =============================================
        # TODO 4: initialize optimizer to zero gradient
        optimizer.zero_grad()
        # =============================================

        output = model(data)
#        print("output ==>")
#        print(output)
#        print("target ==>\n")
#        print(target)

        # =================================================
        # TODO 5: loss -> backpropagation -> update weights
        loss = criterion(output, target)
        # output : (batch_size, 10)   target : (batch_size)

        loss.backward()       # Determine the deviation of loss function w.r.t. weight/bias for each layer
        optimizer.step()      # Update the values of weight/bias for each layer
        # =================================================
        train_acc += calc_acc(output, target)
        train_loss += loss.item()

    train_acc /= len(train_loader)
    train_loss /= len(train_loader)

    return train_acc, train_loss



def validation(model, device, valid_loader, criterion):
    # ===============================
    # TODO 6: switch the model to validation mode
    model.eval()
    # ===============================
    valid_acc = 0.0
    valid_loss = 0.0

    # =========================================
    # TODO 7: turn off the gradient calculation
    with torch.no_grad():
    # =========================================
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            # ================================
            # TODO 8: calculate accuracy, loss
            valid_acc += calc_acc(output, target)
            loss = criterion(output, target)
            valid_loss += loss.item()

            # ================================

    print(len(valid_loader))
    valid_acc /= len(valid_loader)
    valid_loss /= len(valid_loader)

    return valid_acc, valid_loss


def main():
    # ==================
    # TODO 9: set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(torch.__version__)
    print(torch.cuda.is_available())
    # ==================

    # ========================
    # TODO 10: hyperparameters
    # you can add your parameters here
    LEARNING_RATE = 0.05
    MOMENTUM = 0.2
    BATCH_SIZE = 10
    EPOCHS = 60
    TRAIN_DATA_PATH = "./lab2/data/train"
    VALID_DATA_PATH = "./lab2/data/valid"
    MODEL_PATH = "./lab2/model"

    # ========================

    # ===================
    # TODO 11: transforms
    train_transform = transforms.Compose([
        # may be adding some data augmentations?
#        transforms.Resize((224, 224)),
        transforms.GaussianBlur(21, 20),
#        transforms.ColorJitter(brightness=(0.5, 1.2), contrast=(0.5, 1.2)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # ===================
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # =================
    # TODO 12: set up datasets
    # hint: ImageFolder?
    # datasets.ImageFolder will help get the mapping of directory to index (start from 0)
    # For example :
    # {'0': 0, '1': 1, '2': 2, '3': 3}
    train_data = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_transform)
    valid_data = datasets.ImageFolder(root=VALID_DATA_PATH, transform=valid_transform)

    logging.info(f'[DBG] train_data directories : {train_data.classes}')
    logging.info(f'[DBG] train_data mapping :  : {train_data.class_to_idx}')
    logging.info(f'[DBG] valid_data directories : {valid_data.classes}')
    logging.info(f'[DBG] valid_data mapping :  : {valid_data.class_to_idx}')
    # =================

    # ============================
    # TODO 13 : set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
    # ============================

    # build model, criterion and optimizer
    # Why do we need to call train() here???
    model = Net().to(device).train()
    # ================================
    # TODO 14: criterion and optimizer
    # Define loss function and optimizer
    # CrossEntropyLoss is useful when training a classification problem
    # Optimizer is to adjust the hyperparameters per batch data
    criterion = nn.CrossEntropyLoss()  # Define loss function
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)
    # ================================

    # training and validation
    train_acc = [0.0] * EPOCHS
    train_loss = [0.0] * EPOCHS
    valid_acc = [0.0] * EPOCHS
    valid_loss = [0.0] * EPOCHS

    print('Start training...')
    for epoch in range(EPOCHS):
        print(f'epoch {epoch} start...')

        train_acc[epoch], train_loss[epoch] = training(model, device, train_loader, criterion, optimizer)
        valid_acc[epoch], valid_loss[epoch] = validation(model, device, valid_loader, criterion)

        if train_acc[epoch] >= 0.8 and train_acc[epoch] <= 0.85:
            optimizer.param_groups[0]["lr"] = 0.045
            optimizer.param_groups[0]["momentum"] = 0.18
        elif train_acc[epoch] > 0.85 and train_acc[epoch] <= 0.9:
            optimizer.param_groups[0]["lr"] = 0.03
            optimizer.param_groups[0]["momentum"] = 0.06
        elif train_acc[epoch] > 0.9:
            optimizer.param_groups[0]["lr"] = 0.02
            optimizer.param_groups[0]["momentum"] = 0.08
        else:
            optimizer.param_groups[0]["lr"] = 0.05
            optimizer.param_groups[0]["momentum"] = 0.2


        print(f'epoch={epoch} train_acc={train_acc[epoch]} train_loss={train_loss[epoch]} valid_acc={valid_acc[epoch]} valid_loss={valid_loss[epoch]}')
    print('Training finished')

    # ==================================
    # TODO 15: save the model parameters
#    torch.???
    # ==================================

    # ========================================
    # TODO 16: draw accuracy and loss pictures
    # lab2_teamXX_accuracy.png, lab2_teamXX_loss.png
    # hint: plt.plot
    #plt.plot(ep, train_acc)
    plt.plot(train_acc, label="Train")
    plt.plot(valid_acc, label="Validation")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

    # =========================================

if __name__ == '__main__':
    # Set debug level
    # Print logging.info by level=logging.INFO
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
