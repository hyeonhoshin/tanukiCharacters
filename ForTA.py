import torch
import torchvision
from tanukiCharNet import ResNet, BasicBlock
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument('-i', type=str,default = "FtanukiCharNet.pth")

args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
cpu = torch.device('cpu')

model = ResNet(BasicBlock, [3, 4, 6, 3]).to(device)

model.load_state_dict(torch.load(args.i))
model.eval()
batch_size = 8

#Get train Accuracy
train_dir = './abcde'

train_images = dset.ImageFolder(root=train_dir,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),  
                                                    (0.5,0.5,0.5)), 
                           ]))
train_loader = torch.utils.data.DataLoader(train_images, batch_size=batch_size,
                                          shuffle=False, num_workers=8)

print('Number of train images: ', len(train_images))

correct = 0
total = 0
with torch.no_grad():
    for data in train_loader:
        images, labels = data
        images_g, labels_g = images.to(device), labels.to(device)
        outputs = model(images_g)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted.to(cpu) == labels).sum().item()
print('Train Accuracy = {}\n'.format(100 * correct / total))

# Get test Accuracy
test_dir = '../ForTA/abcde'

test_images = dset.ImageFolder(root=test_dir,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),  
                                                    (0.5,0.5,0.5)), 
                           ]))
test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size,
                                          shuffle=False, num_workers=8)

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