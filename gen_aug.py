aug_rate = 100


import cv2
import numpy as np
import os
from PIL import Image
import torch
import cv2

train_labels, train_images = [],[]
train_dir = './abcde'
shape_list = ['a', 'b', 'c', 'd', 'e']

for shape in shape_list:
    print('Getting data for: ', shape)
    for file_name in os.listdir(os.path.join(train_dir,shape)):
        train_images.append(Image.open(os.path.join(train_dir,shape,file_name)))
        #add an integer to the labels list
        train_labels.append(shape_list.index(shape))

print('Number of training images: ', len(train_images))

path_list = ['./aug_abcde/a','./aug_abcde/b','./aug_abcde/c','./aug_abcde/d','./aug_abcde/e']

for path in path_list:
    os.makedirs(path, exist_ok=True)

a_imgs = train_images[0:30]
b_imgs = train_images[30:60]
c_imgs = train_images[60:90]
d_imgs = train_images[90:120]
e_imgs = train_images[120:150]

import PIL.ImageOps
from PIL import Image, ImageEnhance
from random import randint, uniform

def augument(img, pad=2):
    '''
    Input : Pillow image with 3 channel
    Output : Pillow image with 3 channel, and randomly rotated, resized, translated without trimmed reigion.
    '''  

    deg = torch.empty(1).uniform_(-180, 180).item()

    iim = PIL.ImageOps.invert(img)
    iim = np.array(iim)
    iim = cv2.bilateralFilter(iim,9,75,75)
    iim = Image.fromarray(iim)
    iimb = iim.convert('1')

    enh = ImageEnhance.Contrast(iim)
    iim = enh.enhance(torch.empty(1).uniform_(1.0,4.0).item())

    bbox = iimb.getbbox()
    bbox = list(bbox)

    if pad is not 0:
        bbox[0] -= pad
        bbox[1] -= pad
        bbox[2] += pad
        bbox[3] += pad

    iim = iim.crop(bbox)
    iim = iim.rotate(deg, expand=True)

    bbox = iim.getbbox()
    bbox = list(bbox)

    width,height = iim.size
    r = height/width

    if r<1:
        new_width = int(round(torch.empty(1).uniform_(32,349).item()))
        new_height = int(r*new_width)
    else:
        new_height = int(round(torch.empty(1).uniform_(32,349).item()))
        new_width = int(new_height/r)

    iim = iim.resize((new_width, new_height),resample=Image.LANCZOS)

    iim = np.array(iim,dtype=np.uint8)
    iim = cv2.bilateralFilter(iim,9,75,75)

    img = np.zeros((350,350,3),dtype='uint8')

    offset_width = int(round(torch.empty(1).uniform_(0, 350 - iim.shape[0]).item()))
    offset_height = int(round(torch.empty(1).uniform_(0, 350 - iim.shape[1]).item()))

    img[offset_width:offset_width+iim.shape[0], offset_height:offset_height+iim.shape[1]] = iim
    img = 255-img

    return Image.fromarray(img)

# Argument for a
num = 0
for img in a_imgs:
    for i in range(aug_rate-1):
        new_img = augument(img)
        new_img.save("./aug_abcde/a/{}.png".format(num))
        num+=1
    img.save("./aug_abcde/a/{}.png".format(num))

# Argument for b
num = 0
for img in b_imgs:
    for i in range(aug_rate-1):
        new_img = augument(img)
        new_img.save("./aug_abcde/b/{}.png".format(num))
        num+=1
    img.save("./aug_abcde/b/{}.png".format(num))

# Argument for c
num = 0
for img in c_imgs:
    for i in range(aug_rate-1):
        new_img = augument(img)
        new_img.save("./aug_abcde/c/{}.png".format(num))
        num+=1
    img.save("./aug_abcde/c/{}.png".format(num))

# Argument for d
num = 0
for img in d_imgs:
    for i in range(aug_rate-1):
        new_img = augument(img)
        new_img.save("./aug_abcde/d/{}.png".format(num))
        num+=1
    img.save("./aug_abcde/d/{}.png".format(num))

# Argument for e
num = 0
for img in e_imgs:
    for i in range(aug_rate-1):
        new_img = augument(img)
        new_img.save("./aug_abcde/e/{}.png".format(num))
        num+=1
    img.save("./aug_abcde/e/{}.png".format(num))

