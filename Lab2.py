import os
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import logging
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchsummary import summary

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        # ==========================
        # TODO 1: build your network
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout2d(0.1)

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(num_features=128)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(num_features=256)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.conv7 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(num_features=512)
        self.conv8 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv9 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv10 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batch_norm4 = nn.BatchNorm2d(num_features=512)
        self.conv11 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.conv13 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1)
        self.batch_norm5 = nn.BatchNorm2d(num_features=512)

        self.fc1 = nn.Linear(in_features=(512 * 8 * 8), out_features=4096)
        self.fc2 = nn.Linear(in_features=4096, out_features=1792)
        self.fc3 = nn.Linear(in_features=1792, out_features=10)




        # ==========================

    def forward(self, x):
        # (batch_size, 3, 256, 256)

        # ========================
        # TODO 2: forward the data
        # please write down the output size of each layer
        # example:
        # out = self.relu(self.conv1(x))
        # (batch_size, 64, 256, 256)

        out = F.relu(self.conv1(x))  # 224
        out = F.relu(self.conv2(out))
        out = self.batch_norm1(out)
        out = self.drop(out)
        out = self.pool(out)  # 112

        out = F.relu(self.conv3(out))  # 112
        out = F.relu(self.conv4(out))
        out = self.batch_norm2(out)
        out = self.drop(out)
        out = self.pool(out)  # 56

        out = F.relu(self.conv5(out))  # 56
        out = F.relu(self.conv6(out))
        out = F.relu(self.conv7(out))
        out = self.batch_norm3(out)
        out = self.drop(out)
        out = self.pool(out)  # 28

        out = F.relu(self.conv8(out))  # 28
        out = F.relu(self.conv9(out))
        out = F.relu(self.conv10(out))
        out = self.batch_norm4(out)
        out = self.drop(out)
        out = self.pool(out)  # 14

        out = F.relu(self.conv11(out))  # 14
        out = F.relu(self.conv12(out))
        out = F.relu(self.conv13(out))
        out = self.batch_norm5(out)
        out = self.drop(out)
        out = self.pool(out)  # 7

        out = torch.flatten(out, 1)

        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        # ========================
        return out


def calc_acc(output, target):
    predicted = torch.max(output, 1)[1]
    num_samples = target.size(0)
    num_correct = (predicted == target).sum().item()
    return num_correct / num_samples


def training(model, device, train_loader, criterion, optimizer):
    # ===============================
    # TODO 3: switch the model to training mode
    model.train()
    # ===============================
    train_acc = 0.0
    train_loss = 0.0

    for i, (data, target) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, target = data.to(device), target.to(device)

        # =============================================
        # TODO 4: initialize optimizer to zero gradient
        optimizer.zero_grad()
        # =============================================

        output = model(data)

        # =================================================
        # TODO 5: loss -> backpropagation -> update weights
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
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
        for i, (data, target) in tqdm(enumerate(valid_loader), total=len(valid_loader)):
            data, target = data.to(device), target.to(device)

            output = model(data)

            # ================================
            # TODO 8: calculate accuracy, loss
            valid_acc += calc_acc(output, target)
            loss = criterion(output, target)
            valid_loss += loss.item()
            # ================================

    valid_acc /= len(valid_loader)
    valid_loss /= len(valid_loader)

    return valid_acc, valid_loss


def main():
    # ==================
    # TODO 9: set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # ==================

    # ========================
    # TODO 10: hyperparameters
    # you can add your parameters here
    LEARNING_RATE = 0.05
    MOMENTUM = 0.9
    EPOCHS = 10
    BATCH_SIZE = 64
    TRAIN_DATA_PATH = '../data/lab2/data/train/'
    VALID_DATA_PATH = '../data/lab2/data/valid/'
    MODEL_PATH = './trained_result/'

    # ========================

    # ===================
    # TODO 11: transforms
    train_transform = transforms.Compose([
        # may be adding some data augmentations?

        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    # ===================
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    # =================
    # TODO 12: set up datasets
    # hint: ImageFolder?
    train_data = datasets.ImageFolder(root=TRAIN_DATA_PATH, transform=train_transform)
    valid_data = datasets.ImageFolder(root=VALID_DATA_PATH, transform=valid_transform)

    logging.info(f'[DBG] train_data directories : {train_data.classes}')
    logging.info(f'[DBG] train_data mapping :  : {train_data.class_to_idx}')
    logging.info(f'[DBG] valid_data directories : {valid_data.classes}')
    logging.info(f'[DBG] valid_data mapping :  : {valid_data.class_to_idx}')
    # =================

    # ============================
    # TODO 13 : set up dataloaders
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE, num_workers=0, pin_memory=True)
    # ============================

    # build model, criterion and optimizer
    model = Net().to(device).train()
    # ================================
    # TODO 14: criterion and optimizer
    criterion = nn.CrossEntropyLoss()
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

        print(
            f'epoch={epoch} train_acc={train_acc[epoch]} train_loss={train_loss[epoch]} valid_acc={valid_acc[epoch]} valid_loss={valid_loss[epoch]}')
    print('Training finished')

    # ==================================
    # TODO 15: save the model parameters
    #torch.???
    # ==================================

    # ========================================
    # TODO 16: draw accuracy and loss pictures
    # lab2_teamXX_accuracy.png, lab2_teamXX_loss.png
    # hint: plt.plot

    # =========================================

if __name__ == '__main__':
    main()
