import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import torchvision

from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

#create lists to save the labels (the name of the shape)
train_labels, train_images = [],[]
train_dir = './aug_abcde' #"./abcde"
shape_list = ['a', 'b', 'c', 'd', 'e']

batch_size = 16
epochs = 25

import torchvision.datasets as dset
import torchvision.transforms as transforms

train_transform = transforms.Compose([
    transforms.GaussianBlur((5,5), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])
])

trainset = dset.ImageFolder(root=train_dir,
                           transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

val_dir = '../ForTA'

val_images = dset.ImageFolder(root=val_dir,
                           transform=transforms.Compose([
                               #Clearify(pad=2),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),  
                                                    (0.5,0.5,0.5)), 
                           ]))
valloader = torch.utils.data.DataLoader(val_images, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# ## Generate Model
# * Resnet50 based

import torch.nn as nn
import math

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=5, input_channels=3):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout2d(p=0.5,inplace=True)

        #print "block.expansion=",block.expansion
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.dropout(x)
        #print "avepool: ",x.data.shape
        x = x.view(x.size(0), -1)
        #print "view: ",x.data.shape
        x = self.fc(x)

        return x


# ### Model and Learning environment

criterion = nn.CrossEntropyLoss()
learning_rate = 1e-3 # Adaptive learning rate
lr_decay_rate = 0.95

model = ResNet(BasicBlock, [3, 4, 6, 3]).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

#sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0 =10, T_mult=2, eta_min=0, last_epoch=-1)
sch = torch.optim.lr_scheduler.StepLR(optimizer=optimizer ,step_size=1, gamma=lr_decay_rate)

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)
        
model.apply(init_weights)


# ## Training
# - Use only the splited dataset

itr = 0
for epoch in range(epochs):  # loop over the dataset multiple times
    #print('------- Epoch:', epoch,'LR:', sch.get_lr(),'-------')
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        x = inputs.to(device)
        y = labels.to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if itr%10==9:
            print('[%d, %5d] loss: %.6f' %(epoch + 1, i + 1, running_loss / batch_size))
        itr += 1
        writer.add_scalar('training_loss', running_loss / batch_size, itr+ 1)
        running_loss = 0.0
        
    # Testing
    model.eval()
    correct = 0
    total = 0
    cpu = torch.device('cpu')
    
    with torch.no_grad():
        for data in valloader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to(cpu) == labels).sum().item()

    print('Validation accuracy: {:.3f}%'.format(100 * correct / total))
    
    writer.add_scalar('val_acc', 100 * correct / total, epoch+ 1)
    
    correct = 0
    total = 0
    with torch.no_grad():
        for data in trainloader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to(cpu) == labels).sum().item()

    print('Train accuracy: {:.3f}%'.format(100 * correct / total))
    writer.add_scalar('train_acc', 100 * correct / total, epoch+ 1)
            
    # Learning rate changes
    sch.step()
    writer.flush()
    
    torch.save(model.state_dict(), "F_tanukiChar_epoch{}_GBlur+SSDAug+LRdecay0.95.pth".format(epoch+1))

print('Finished Training')
writer.close()


# ## For TA

'''
model.load_state_dict(torch.load(fname))
model.eval()

test_dir = '../ForTA/abcde'

test_images = dset.ImageFolder(root=test_dir,
                           transform=transforms.Compose([
                               Clearify(pad=2),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),  
                                                    (0.5,0.5,0.5)), 
                           ]))
test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

print('Number of test images: ', len(test_images))

correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.to(cpu) == labels).sum().item()

print('Test Accuracy = {}'.format(100 * correct / total))
'''
