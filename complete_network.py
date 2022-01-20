import torchvision
import torch
import torchvision.transforms as transforms
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL.Image import Image

from mean_and_std import get_mean
from mean_and_std import get_std
# imports for training
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
num_classes = 5

train_dataset_path = './data/train'
test_dataset_path = './data/test'
training_transforms = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
                                          transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
training_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=training_transforms)
train_loader = torch.utils.data.DataLoader(dataset=training_dataset, batch_size=32, shuffle=False)
# print("mean:", get_mean(train_loader))
# print("std:", get_std(train_loader))
#mean = get_mean(train_loader)
#std = get_std(train_loader)

classes = ['0', '1', '2', '3', '4']

train_transforms = transforms.Compose([
   transforms.Resize((224, 224)),
   transforms.RandomHorizontalFlip(),
   transforms.RandomRotation(10),
   transforms.ToTensor(),
   transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
])

train_dataset = torchvision.datasets.ImageFolder(root=train_dataset_path, transform=train_transforms)
test_dataset = torchvision.datasets.ImageFolder(root=test_dataset_path, transform=test_transforms)


#def show_transformed_images(dataset):
#    loader = torch.utils.data.DataLoader(dataset, batch_size=6, shuffle=True)
#    batch = next(iter(loader))
#    images, labels = batch
#
#    grid = torchvision.utils.make_grid(images, nrow=3)
#    plt.figure(figsize=(11, 11))
#    plt.imshow(np.transpose(grid, (1, 2, 0)))
#    print("labels:", labels)


#show_transformed_images(train_dataset)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False) # Mini Batch Greadient escent Algorithm

# ARCHITECTURE


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
        self.expansion = 4
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU()
        self.i_downsample = i_downsample
        self.stride = stride


    def forward(self,x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, Block, layers, image_channels,num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(Block, layers[0], out_channels=64, stride=2)
        self.layer2 = self._make_layer(Block, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(Block, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(Block, layers[3], out_channels=512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * 4, num_classes)

    def forward(self,x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x


    def _make_layer(self,block, num_residual_blocks, out_channels, stride):
        i_downsample = None
        layers = []
        if stride != 1 or self.in_channels != out_channels * 4:
            i_downsample = nn.Sequential(nn.Conv2d(self.in_channels, out_channels*4, kernel_size=1, stride=stride,
                                                   bias=False), nn.BatchNorm2d(out_channels*4))

        layers.append(block(self.in_channels,out_channels,i_downsample, stride))

        self.in_channels = out_channels * 4

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, out_channels))


        return nn.Sequential(*layers)




def ResNet50(img_channel=3, num_classes=1000):
    return ResNet(Block, [3,4,6,3], img_channel, num_classes )


def ResNet101(img_channel=3, num_classes=1000):
    return ResNet(Block, [3,4,23,3], img_channel, num_classes)


def ResNet152(img_channel=3, num_classes=1000):
    return ResNet(Block, [3,8,36,3], img_channel, num_classes)



# training neural network



def set_device():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"

    return torch.device(dev)


def train_nn(model, train_loader, test_loader, criterion, optimizer, n_epochs):
    device = set_device()

    for epoch in range(n_epochs):
        print("Epoch number is :", (epoch+1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        best_acc = 0.0

        for data in train_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            optimizer.zero_grad()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100 * (running_correct/total)

        print("  - Training dataset.Got {} out of {} images correctly {}. Epoch loss :{}" .format(running_correct,
                                                                                                total, epoch_acc,
                                                                                                   epoch_loss))

        test_data_acc = evaluate_model_on_test_set(model, test_loader)

        if test_data_acc > best_acc:
            best_acc = test_data_acc
            save_checkpoint(model, epoch, optimizer, best_acc)


    print("Finished")
    return model


def evaluate_model_on_test_set(model, test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device = set_device()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            total += labels.size(0)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            predicted_correctly_on_epoch += (predicted == labels).sum().item()

    epoch_acc = 100 * (predicted_correctly_on_epoch / total)
    print("  - TEST dataset.Got {} out of {} images correctly {}.".format(predicted_correctly_on_epoch,
                                                                                             total, epoch_acc,
                                                                                             ))
    return epoch_acc


def save_checkpoint(model, epoch, optimizer, best_acc):
    state = {
        'epoch': epoch+1,
        'model': model.state_dict(),
        'best_accuracy': best_acc
    }
    torch.save(state, 'best_model')





resnet18_model = ResNet(Block,[3,4,23,3],image_channels=3, num_classes=5)
device = set_device()
resnet_18_model = resnet18_model.to(device)
loss_fn = nn.CrossEntropyLoss()

optimizer = optim.SGD(resnet18_model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.003)



train_nn(resnet18_model, train_loader, test_loader, loss_fn, optimizer, 1)

model_sample = torch.load('best_model')

image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((-0.1297,-0.2573,-0.3357), (0.4444,0.4226,0.4164))
])


def classify(model, image_transforms, image_path, classes):
    model = model.eval()
    image = Image.open(image_path)
    image = image_transforms(image.float())
    image = image.unsqueeze(0)
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    print(classes[predicted.item()])


classify(model_sample, image_transforms, "./data/val/0.png", classes)



#model = torch.load('best_model.pth')

#image_transforms = transforms.Compose([
#    transforms.Resize((224, 224)),
#    transforms.ToTensor(),
#    transforms.Normalize(torch.Tensor(mean), torch.Tensor(std))
#])


#def classify(model, image_transforms, image_path, classes):
 #   model = model.eval()
#    image = Image.open(image_path)
#    image = image_transforms(image.float())
#    image = image.unsqueeze(0)
#    output = model(image)
#    _, predicted = torch.max(output.data, 1)
 #   print(classes[predicted.item()])
###




