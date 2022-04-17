import numpy as np
from PIL import Image, ImageFilter
import torch
import torchvision.transforms.functional as TF

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def quantize_to_int(x, q=8):
    """ Make interger divisible by q, but never smaller than q. """    
    return int(q) if x<q else int(np.ceil(x/q)*q)

class AddGaussianNoise(torch.nn.Module):
    def __init__(self, std=0.01):
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std
    
    def __repr__(self):
        return self.__class__.__name__ + f'(std={self.std})'

class RandomGaussianBlur(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(np.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = img.filter(ImageFilter.GaussianBlur(radius=abs(np.random.normal(loc=0.0, scale=0.1))))
        return img

class RandomGamma(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(np.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = TF.adjust_gamma(img, gamma=np.random.normal(loc=1.0, scale=0.03))
        return img

class RandomBrightness(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(np.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = TF.adjust_brightness(img, brightness_factor=np.random.normal(loc=1.0, scale=0.01))
        return img

class RandomSharpness(torch.nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = float(np.clip(p, 0.0, 1.0))

    def forward(self, img):
        if np.random.uniform() < self.p:
            img = img.filter(ImageFilter.UnsharpMask(radius=max(np.random.normal(loc=0.0, scale=0.03), 0.0)))
        return img

class CustomTransformation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.gaussian = RandomGaussianBlur(p=1.0)
        self.gamma = RandomGamma(p=1.0)
        self.brightness = RandomBrightness(p=1.0)
        self.sharpen = RandomSharpness(p=1.0)

    def forward(self, img):
        img = self.gaussian(img)
        # img = self.gamma(img)
        # img = self.brightness(img)
        # img = self.sharpen(img)
        return img
