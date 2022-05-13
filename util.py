import os
import time

import numpy as np
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.transforms.functional as TF


color_options = [
    'white', 'black', 'gray', 'red', 'blue',
    'green', 'yellow', 'purple', 'brown', 'orange'
]
shape_options = [
    "circle", "semicircle", "quartercircle", "triangle", "square",
    "rectangle", "trapezoid", "pentagon", "hexagon", "heptagon",
    "octagon", "star", "cross"
]
# No W or 9
letter_options = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
    'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'X', 'Y', 'Z', '1', '2', '3', '4', '5',
    '6', '7', '8', '0'
]

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def load_backgrounds(folder_path, size=None):
    if folder_path==None: return None
    print(" * Loading backgrounds...")
    ts = time.time()
    backgrounds = [pil_loader(os.path.join(os.getcwd(), x)) for x in os.scandir(folder_path)]
    if size is not None:
        backgrounds = [T.Resize(size)(img) for img in backgrounds]
    print(f" ** Backgrounds loaded. {time.time()-ts:.03f} seconds for {len(backgrounds)} images.")
    return backgrounds

def quantize_to_int(x, q=8):
    """ Make interger divisible by q, but never smaller than q. """    
    return int(q) if x<q else int(np.floor(x/q)*q)

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

def pixel_accuracy(preds, targets, threshold: float = 0.5):
    accsum = 0.0
    preds = preds > threshold
    correct = (preds == targets).sum()
    total = targets.shape[0]*targets.shape[2]*targets.shape[3]
    return correct/total

def jaccard_iou(preds, targets, smooth: float = 1.0):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    total = (preds + targets).sum()
    union = total - intersection
    return (intersection + smooth) / (union + smooth)

def dice_coeff(preds, targets, smooth: float = 1.0):
    preds = preds.view(-1)
    targets = targets.view(-1)
    intersection = (preds * targets).sum()
    unionset = (preds + targets).sum()
    return (2.0 * intersection + smooth) / (unionset + smooth)

def tversky_measure(preds, targets, alpha: float = 0.5, beta: float = 0.5, smooth: float = 1.0):
    preds = preds.view(-1)
    targets = targets.view(-1)
    tp = (preds * targets).sum()
    fp = ((1.0-targets) * preds).sum()
    fn = (targets * (1.0-preds)).sum()
    return (tp + smooth) / (tp + alpha*fp + beta*fn + smooth)

def focal_metric(logits, targets, alpha: float, gamma: float):
    logits = logits.view(-1)
    targets = targets.view(-1)
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    bce_exp = torch.exp(-bce)
    return alpha * bce * (1.0-bce_exp)**gamma
