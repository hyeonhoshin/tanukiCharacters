import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
from tanukiCharNet import BasicBlock, ResNet
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn as nn
from tanukiDataAug import Augument

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

cpu = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
batch_size = 8
epochs = 35

# Data from augumented dset
aug_transform = transforms.Compose([

    transforms.GaussianBlur((5,5), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])
])
augset = dset.ImageFolder(root='./aug_abcde',
                           transform=aug_transform)
augloader = torch.utils.data.DataLoader(augset, batch_size=batch_size,
                                          shuffle=True, num_workers=8)

# Data from train dset
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])
])
                                    
trainset = dset.ImageFolder(root='./abcde',
                           transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=False, num_workers=8)

# Data from val dset
val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])
])
                                    
valset = dset.ImageFolder(root='../ForTA/abcde',
                           transform=val_transform)
valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size,
                                          shuffle=False, num_workers=8)

### Model and Learning environment
criterion = nn.CrossEntropyLoss()
learning_rate = 2e-3
lr_decay_rate = 0.95

model = ResNet(BasicBlock, [3, 4, 6, 3]).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
sch = torch.optim.lr_scheduler.StepLR(optimizer=optimizer ,step_size=1, gamma=lr_decay_rate)

## Initialize weights
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
model.apply(init_weights)

## Training
itr = 0
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(augloader, 0):
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

        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        # print statistics
        running_loss += loss.item()
        if i%4==3:
            print('[%d, %5d] loss: %.6f' %(epoch + 1, i + 1, running_loss / batch_size))
            running_loss = 0.0
        
        # writing statistics
        writer.add_scalar('training_loss', running_loss / batch_size, itr+1)

    print('Train accuracy: {:.3f}%'.format(100 * correct / total))
        
    # Get val Accuracy
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data in valloader:
            images, labels = data
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted.to(cpu) == labels).sum().item()

    print('Val accuracy: {:.3f}%'.format(100 * correct / total))
    writer.add_scalar('val_acc', 100 * correct / total, epoch+ 1)
            
    # Learning rate changes
    writer.flush()
    sch.step()
    
    torch.save(model.state_dict(), "F_tanukiChar_epoch{}_GBlur+SSDAug+LRdecay0.95.pth".format(epoch+1))

print('Finished Training')
writer.close()
