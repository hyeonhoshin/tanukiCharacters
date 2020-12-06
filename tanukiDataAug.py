import PIL.ImageOps
from PIL import Image, ImageEnhance
from random import randint, uniform
import torch
import numpy as np
import cv2

class Augument(torch.nn.Module):
    '''
    Input : Pillow image with 3 channel
    Output : Pillow image with 3 channel, and randomly rotated, resized, translated without trimmed reigion.
    '''  

    def __init__(self, pad=1):
        super().__init__()
        self.pad = pad
        pass

    def __call__(self, img):
        pad = self.pad

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
    
class Clearify(torch.nn.Module):
    '''
    Input : Pillow image with 3 channel
    Output : Pillow image with 3 channel, and randomly rotated, resized, translated without trimmed reigion.
    '''  

    def __init__(self, pad=1):
        super().__init__()
        self.pad = pad
        pass

    def __call__(self, img):
        pad = self.pad

        iim = PIL.ImageOps.invert(img)

        iim = np.array(iim,dtype=np.uint8)
        iim[iim<20] = 0
        iim = Image.fromarray(iim)

        enh = ImageEnhance.Contrast(iim)
        iim = enh.enhance(1.5)

        iimb = np.array(iim)
        iimb[iimb<125] = 0
        iimb = Image.fromarray(iimb)

        bbox = iimb.getbbox()
        bbox = list(bbox)

        if pad is not 0:
            bbox[0] -= pad
            bbox[1] -= pad
            bbox[2] += pad
            bbox[3] += pad

        iim = iim.crop(bbox)
        width,height = iim.size
        r = height/width

        if r<1:
            new_width = 348
            new_height = int(r*new_width)
        else:
            new_height = 348
            new_width = int(new_height/r)

        iim = iim.resize((new_width, new_height))

        iim = np.array(iim,dtype='uint8')

        img = np.zeros((350,350,3),dtype='uint8')

        offset_width = np.random.randint(0, 350 - iim.shape[0])
        offset_height = np.random.randint(0, 350 - iim.shape[1])

        img[offset_width:offset_width+iim.shape[0], offset_height:offset_height+iim.shape[1]] = iim
        img = 255-img

        return Image.fromarray(img)