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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device('cpu')

# create lists to save the labels (the name of the shape)
train_labels, train_images = [],[]
train_dir = "./abcde" #'./aug_abcde' when using dset from gen_aug.py
shape_list = ['a', 'b', 'c', 'd', 'e']

batch_size = 8
epochs = 70*100 #Epochs * Augument rate. Same with 70epochs with gen_aug

# Data augumentation
train_transform = transforms.Compose([
    Augument(pad=1),
    transforms.GaussianBlur((5,5), sigma=(0.1, 2.0)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],
                         std=[0.5,0.5,0.5])
])

trainset = dset.ImageFolder(root=train_dir,
                           transform=train_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

### Model and Learning environment
criterion = nn.CrossEntropyLoss()
learning_rate = 2e-3
lr_decay_rate = 0.99948 # 0.99948^3500 = 0.95^35

model = ResNet(BasicBlock, [3, 4, 6, 3]).to(device)
optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
sch = torch.optim.lr_scheduler.StepLR(optimizer=optimizer ,step_size=1, gamma=lr_decay_rate)

## Initialize weights
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        
model.apply(init_weights)

itr = 0
## Training
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    total = 0
    correct = 0
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

        _, predicted = outputs.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        # print statistics
        running_loss += loss.item()
        if i%10==9:
            print('[%d, %5d] loss: %.6f' %(epoch + 1, i + 1, running_loss / batch_size))
            running_loss = 0.0
        
        # writing statistics
        writer.add_scalar('training_loss', running_loss / batch_size, itr+1)

    print('Train accuracy: {:.3f}%'.format(100 * correct / total))
    writer.add_scalar('train_acc', 100 * correct / total, epoch+ 1)
            
    # Learning rate changes
    sch.step()
    
    if epoch%10 == 10-1:
        torch.save(model.state_dict(), "F_tanukiChar_epoch{}.pth".format(epoch+1))

print('Finished Training')
writer.close()
