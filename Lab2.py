import os

import numpy
import numpy as np
import torch
import torch.nn as nn
import torchvision.utils
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
        self.drop = nn.Dropout2d(0.075)

        self.conv0 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=35, stride=1, padding=1)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=72, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=72, out_channels=72, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=72)

        self.conv3 = nn.Conv2d(in_channels=72, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=64)

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(num_features=64)

        self.conv8 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv9 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(num_features=128)

        self.conv11 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv12 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv13 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(num_features=128)

        self.fc1 = nn.Linear(in_features=(128 * 8 * 8), out_features=1024)
        self.fc2 = nn.Linear(in_features=1024, out_features=10)
        self.fc3 = nn.Linear(in_features=128, out_features=10)
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

        #out = self.relu(self.conv0(x))
        out = self.relu(self.conv1(x))  # 256
        out = self.relu(self.conv2(out))
        out = self.batch_norm1(out)
        out = self.drop(out)
        out = self.pool(out)  # 128

        out = self.relu(self.conv3(out))  # 128
        out = self.relu(self.conv4(out))
        out = self.batch_norm2(out)
        out = self.drop(out)
        out = self.pool(out)  # 64

        out = self.relu(self.conv5(out))  # 64
        out = self.relu(self.conv6(out))
        #out = self.relu(self.conv7(out))
        #out = self.batch_norm3(out)
        #out = self.drop(out)
        out = self.pool(out)  # 32

        out = self.relu(self.conv8(out))  # 32
        #out = self.relu(self.conv9(out))
        #out = self.relu(self.conv10(out))
        #out = self.batch_norm4(out)
        #out = self.drop(out)
        out = self.pool(out)  # 16

        out = self.relu(self.conv11(out))  # 16
        #out = self.relu(self.conv12(out))
        #out = self.relu(self.conv13(out))
        out = self.pool(out)  # 8

        out = torch.flatten(out, 1)

        out = self.relu(self.fc1(out))
        out = self.fc2(out)
        # ========================
        return out


def calc_acc(output, target, mode):
    predicted = torch.max(output, 1)[1]
    num_samples = target.size(0)
    num_correct = (predicted == target).sum().item()

    #if mode == "train":
    #    print("predicted >>")
    #    print(predicted)
    #    print("target >>")
    #    print(target)

    return num_correct / num_samples

def training(model, device, train_loader, criterion, optimizer):
    # ===============================
    # TODO 3: switch the model to training mode
    # Actually, we can ignore this setting because when self.training=True by default
    model.train()
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

        # =================================================
        # TODO 5: loss -> backpropagation -> update weights
        loss = criterion(output, target)
        # output : (batch_size, 10)   target : (batch_size)

        loss.backward()       # Determine the deviation of loss function w.r.t. weight/bias for each layer
        optimizer.step()      # Update the values of weight/bias for each layer
        # =================================================
        train_acc += calc_acc(output, target, "train")
        train_loss += loss.item()

    train_acc /= len(train_loader)
    train_loss /= len(train_loader)

    return train_acc, train_loss

def validation(model, device, valid_loader, criterion, batch_size):
    # ===============================
    # TODO 6: switch the model to validation mode
    model.train(False)
    model.training = False
    # ===============================
    valid_acc = 0.0
    valid_loss = 0.0

    # =========================================
    # TODO 7: turn off the gradient calculation
    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)

            # ================================
            # TODO 8: calculate accuracy, loss
            valid_acc += calc_acc(output, target, "valid")
            loss = criterion(output, target)
            valid_loss += loss.item()

            # Show images
            #grid = torchvision.utils.make_grid(data)
            #images = torchvision.transforms.ToPILImage()(grid)
            #images.show()
            # ================================

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
    LEARNING_RATE = 0.01
    MOMENTUM = 0.5
    BATCH_SIZE = 32
    EPOCHS = 30
    TRAIN_DATA_PATH = '../data/lab2/data/train/'
    VALID_DATA_PATH = '../data/lab2/data/valid/'
    MODEL_PATH = "./lab2/model"

    # ========================

    # ===================
    # TODO 11: transforms
    train_transform = transforms.Compose([
        # may be adding some data augmentations?
        #ransforms.Resize((224, 224)),
        transforms.GaussianBlur(21),
        transforms.RandomAffine(degrees=20, shear=(0, 0, 0, 45)),
        transforms.ColorJitter(brightness=(0.1), contrast=(1.5), hue=(0.5)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # ===================
    valid_transform = transforms.Compose([
        #transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # =================
    # TODO 12: set up datasets
    # hint: ImageFolder?
    # datasets.ImageFolder will help get the mapping of directory to index (start from 0)
    # For example :
    # {'0': 0, '1': 1, '2': 2, '3': 3}
    train_data = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=valid_transform)
    valid_data = datasets.ImageFolder(root=VALID_DATA_PATH, transform=valid_transform)

    logging.info(f'[DBG] train_data directories : {train_data.classes}')
    logging.info(f'[DBG] train_data mapping :  : {train_data.class_to_idx}')
    logging.info(f'[DBG] valid_data directories : {valid_data.classes}')
    logging.info(f'[DBG] valid_data mapping :  : {valid_data.class_to_idx}')
    # =================

    # ============================
    # TODO 13 : set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, num_workers=2, pin_memory=True)
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
        valid_acc[epoch], valid_loss[epoch] = validation(model, device, valid_loader, criterion, BATCH_SIZE)

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
    plt.plot(train_acc, label="Train")
    plt.plot(valid_acc, label="Validation")
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()
    plt.plot(train_loss, label="Train")
    plt.plot(valid_loss, label="Validation")
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

    # =========================================

if __name__ == '__main__':
    # Set debug level
    # Print logging.info by level=logging.INFO
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    main()
