import torch

import torch.nn as nn

import torch.optim as optim

import torchvision.transforms as transforms

import torchvision.datasets as datasets

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform_train = transforms.Compose([

    transforms.RandomRotation(40),

    transforms.RandomHorizontalFlip(),

    transforms.RandomResizedCrop(150),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])

transform_test = transforms.Compose([

    transforms.Resize(150),

    transforms.ToTensor(),

    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])

train_data = datasets.ImageFolder('Masks Dataset/Train', transform=transform_train)

valid_data = datasets.ImageFolder('Masks Dataset/Validation', transform=transform_test)

test_data = datasets.ImageFolder('Masks Dataset/Test', transform=transform_test)

train_loader = DataLoader(train_data, batch_size=30, shuffle=True)

valid_loader = DataLoader(valid_data, batch_size=30, shuffle=False)

test_loader = DataLoader(test_data, batch_size=30, shuffle=False)

model = nn.Sequential(

    nn.Conv2d(3, 32, kernel_size=3, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Dropout(0.5),

    nn.Conv2d(32, 64, kernel_size=3, padding=1),

    nn.ReLU(),

    nn.MaxPool2d(2),

    nn.Dropout(0.5),

    nn.Flatten(),

    nn.Linear(64*18*18, 256),

    nn.ReLU(),

    nn.Dropout(0.5),

    nn.Linear(256, 1),

    nn.Sigmoid()

)

model.to(device)

criterion = nn.BCELoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 40

for epoch in range(epochs):

    running_loss = 0.0

    running_corrects = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(device)

        labels = labels.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):

            outputs = model(inputs)

            loss = criterion(outputs, labels.float().unsqueeze(1))

            _, preds = torch.max(outputs, 1)

            loss.backward()

            optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        running_corrects += torch.sum(preds == labels.data)

    epoch_loss = running_loss / len(train_data)

    epoch_acc = running_corrects.double() / len(train_data)

    valid_loss = 0.0

    valid_corrects = 0

    with torch.no_grad():

        for inputs, labels in valid_loader:

            inputs = inputs.to(device)

            labels = labels.to(device)

            outputs = model(inputs)

            loss = criterion(outputs, labels.float().unsqueeze(1))

            _, preds = torch.max(outputs, 1)

            valid_loss += loss.item() * inputs.size(0)

            valid_corrects += torch.sum(preds == labels.data)

    valid_loss /= len(valid_data)

    valid_acc = valid_corrects.double() / len(valid_data)

    print('Epoch [{}/{}], train_loss: {:.4f}, train_acc: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}'

          .format(epoch+1, epochs, epoch_loss
test_loss = 0

test_acc = 0

steps = 0

with torch.no_grad():

for batch_idx, (x_test, y_test) in enumerate(test_generator):

steps += 1

model.eval()

x_test, y_test = x_test.to(device), y_test.to(device)

outputs = model(x_test)

loss = criterion(outputs, y_test.unsqueeze(1))
