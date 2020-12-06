import torch
import torchvision
from tanukiCharNet import ResNet, BasicBlock
import torchvision.datasets as dset
import torchvision.transforms as transforms
import argparse

parser = argparse.ArgumentParser(description='Trainer')
parser.add_argument('-i', type=str,default = "FtanukiCharNet.pth")

args = parser.parse_args()

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
cpu = torch.device('cpu')

model = ResNet(BasicBlock, [3, 4, 6, 3]).to(device)

model.load_state_dict(torch.load(args.i))
model.eval()
batch_size = 16

test_dir = '../ForTA/abcde'

test_images = dset.ImageFolder(root=test_dir,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.5,0.5,0.5),  
                                                    (0.5,0.5,0.5)), 
                           ]))
test_loader = torch.utils.data.DataLoader(test_images, batch_size=batch_size,
                                          shuffle=True, num_workers=4)

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