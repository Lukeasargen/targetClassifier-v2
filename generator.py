import os
import random
import time
import numpy as np

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import psutil
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from util import load_backgrounds
from util import AddGaussianNoise, CustomTransformation

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'  # draw.polygon raised an error

def rotate_2d(vector, angle, degrees = False):
    """
    Rotate a 2d vector counter-clockwise by @angle.\n
    vector  : [x, y]\n
    angle   : rotation angle
    degrees : set True if angle is in degrees, default is False. 
    """
    if degrees:
        angle = np.radians(angle)
    v = np.array(vector)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array(((c, -s), (s, c)))
    return v.dot(R)

def color_to_hsv(color):
    """ Return a string that is used by PIL to specify HSL colorspace """
    options = {
        'red': lambda: (np.random.randint(0, 4), np.random.randint(50, 100), np.random.randint(40, 60)),
        'orange': lambda: (np.random.randint(9, 33), np.random.randint(50, 100), np.random.randint(40, 60)),
        'yellow': lambda: (np.random.randint(43, 55), np.random.randint(50, 100), np.random.randint(40, 60)),
        'green': lambda: (np.random.randint(75, 120), np.random.randint(50, 100), np.random.randint(40, 60)),
        'blue': lambda: (np.random.randint(200, 233), np.random.randint(50, 100), np.random.randint(40, 60)),
        'purple': lambda: (np.random.randint(266, 291), np.random.randint(50, 100), np.random.randint(40, 60)),
        'brown': lambda: (np.random.randint(13, 20), np.random.randint(25, 50), np.random.randint(22, 40)),
        'black': lambda: (np.random.randint(0, 360), np.random.randint(0, 12), np.random.randint(0, 6)),
        'gray': lambda: (np.random.randint(0, 360), np.random.randint(0, 12), np.random.randint(25, 60)),
        'white': lambda: (np.random.randint(0, 360), np.random.randint(0, 12), np.random.randint(80, 100))
    }
    return 'hsl(%d, %d%%, %d%%)' % options[color]()

def make_regular_polygon(radius, sides, angle, center=(0,0)):
    """ Helper function that returns a list of tuples in a regular polygon
        which are centered at the center argument. """
    step = 2*np.pi / sides
    points = []
    for i in range(sides):
        points.append( (radius*np.cos((step*i) + np.radians(angle))) + center[0] )
        points.append( (radius*np.sin((step*i) + np.radians(angle))) + center[1] )
    return points

class TargetGenerator():
    def __init__(self, img_size, min_size, alias_factor=1, target_transforms=None, backgrounds=None):
        self.img_size = (img_size,img_size) if type(img_size)!=tuple else img_size
        self.min_size = min_size
        self.alias_factor = alias_factor
        self.target_transforms = target_transforms
        self.backgrounds = backgrounds
        if self.backgrounds is not None:
            self.bkg_count = 0  # track which background index to get next
            self.bkg_idxs = list(range(len(self.backgrounds)))  # shuffled list of indexes
        
        self.color_options = [
            'white', 'black', 'gray', 'red', 'blue',
            'green', 'yellow', 'purple', 'brown', 'orange'
        ]
        self.shape_options = [
            "circle", "semicircle", "quartercircle", "triangle", "square",
            "rectangle", "trapezoid", "pentagon", "hexagon", "heptagon",
            "octagon", "star", "cross"
        ]
        # No W or 9
        self.letter_options = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
            'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
            'U', 'V', 'X', 'Y', 'Z', '1', '2', '3', '4', '5',
            '6', '7', '8', '0'
        ]

        # Compute the output size. 2 colors, 1 shape, 1 letter, 2 values for orientaion
        self.output_sizes = {
            "has_target": 1,
            "angle": 2,
            "shape": len(self.shape_options),
            "letter": len(self.letter_options),
            "shape_color": len(self.color_options),
            "letter_color": len(self.color_options),
        }
        self.num_outputs = sum(self.output_sizes.values())

    def draw_shape(self, draw, img_size, min_size, shp_idx, shp_color_idx):
        """ Do not use directly. This is called within draw_target.
            This function draws the specified shape and color.
            Scale and rotation are uniformly sampled.
            It returns values that specify how to draw the letter.
        """
        shape_color = color_to_hsv(self.color_options[shp_color_idx])
        shape = self.shape_options[shp_idx]
        # Uniformly sample that target size.
        # Half this is the radius of the circumscribed circle. Polygon vertices are on this circle.
        r = (np.random.uniform(min_size*self.alias_factor, min(self.alias_factor*img_size[0], self.alias_factor*img_size[1]))) // 2
        cx, cy = np.random.randint(r, self.alias_factor*img_size[0]-r), np.random.randint(r, self.alias_factor*img_size[1]-r)
        # Rotate each shape randomly.
        angle = np.random.uniform(0, 360)
        # Each shape as a different ltr_size. This defines an inscribed circle in the shape.
        if shape == "circle":
            ltr_size = int(r*np.random.randint(50, 75) / 100)
            top = (cx-r, cy-r); bot = (cx+r, cy+r)
            draw.pieslice([top, bot], 0, 360, fill=shape_color)
        elif shape == "quartercircle":
            # slice is in the bottom right, so shift top right rotated by angle
            rr = 2*r/np.sqrt(2)  # outer circle radius that fits the quarter circle
            ss = rr / (1+np.sqrt(2))  # inner circle radius
            ltr_size = int(ss*np.random.randint(75, 90) / 100)
            sx, sy = np.sqrt(2)*ss*np.cos(np.radians(-angle+45)), np.sqrt(2)*ss*np.sin(np.radians(-angle+45))
            top = (cx-(rr)-sx, cy-(rr)-sy)
            bot = (cx+(rr)-sx, cy+(rr)-sy)
            draw.pieslice([top, bot], 0-angle, 90-angle, fill=shape_color)
        elif shape == "semicircle":
            # slice is in the bottom, so shift up rotated by angle
            rr = r / np.sqrt(5/4)  # outer circle radius that fits the semi circle
            ltr_size = int(0.5*rr*np.random.randint(70, 90) / 100)
            sx, sy = 0.5*rr*np.sin(np.radians(angle)), 0.5*rr*np.cos(np.radians(angle))
            top = (cx-(rr)+sx, cy-(rr)+sy)
            bot = (cx+(rr)+sx, cy+(rr)+sy)
            draw.pieslice([top, bot], 180-angle, -angle, fill=shape_color)
        elif shape == "square":
            radius = r*np.random.randint(85, 100) / 100
            ltr_size = int( ( r*np.random.randint(70, 90) ) / ( 100*np.sqrt(2) ) )
            points = make_regular_polygon(radius, 4, -angle, center=(cx, cy))
            draw.polygon(points, fill=shape_color)
        elif shape == "triangle":
            radius = r*np.random.randint(85, 100) / 100
            ltr_size = int(radius*np.random.randint(40, 50) / 100)
            points = make_regular_polygon(radius, 3, -angle, center=(cx, cy))
            draw.polygon(points, fill=shape_color)
        elif shape == "rectangle":
            h = int(r*np.random.randint(81, 97) / 100)
            w = np.sqrt(r*r-h*h)
            ltr_size = int(min(w, h)*np.random.randint(85, 96) / 100)
            points = [(+w,+h),(+w,-h),(-w,-h),(-w,+h)]
            points = rotate_2d(points, angle, degrees=True)
            points = [(x[0]+cx, x[1]+cy) for x in points]
            draw.polygon(points, fill=shape_color)
        elif shape == "pentagon":
            radius = r*np.random.randint(80, 100) / 100
            ltr_size = int(radius*np.random.randint(50, 75) / 100)
            b = -90/5 + np.random.choice([0, 180])
            points = make_regular_polygon(radius, 5, -angle+b, center=(cx, cy))
            draw.polygon(points, fill=shape_color)
        elif shape == "hexagon":
            radius = r*np.random.randint(80, 100) / 100
            ltr_size = int(radius*np.random.randint(50, 80) / 100)
            b = np.random.choice([0, 30])
            points = make_regular_polygon(radius, 6, -angle+b, center=(cx, cy))
            draw.polygon(points, fill=shape_color)
        elif shape == "heptagon":
            radius = r*np.random.randint(80, 100) / 100
            ltr_size = int(radius*np.random.randint(50, 85) / 100)
            points = make_regular_polygon(radius, 7, -angle, center=(cx, cy))
            draw.polygon(points, fill=shape_color)
        elif shape == "octagon":
            radius = r*np.random.randint(80, 100) / 100
            ltr_size = int(radius*np.random.randint(50, 85) / 100)
            points = make_regular_polygon(radius, 8, -angle, center=(cx, cy))
            draw.polygon(points, fill=shape_color)
        elif shape == "cross":
            h = int(r*np.random.randint(35, 43) / 100)
            w = np.sqrt(r*r - h*h)
            ltr_size = int(min(w, h)*np.random.randint(92, 99) / 100)
            b = np.random.choice([0, 45])
            points = [( +w, +h ),( +w, -h ),( -w, -h ),( -w, +h )]
            points1 = [(x[0]+cx, x[1]+cy) for x in rotate_2d(points, angle+b, degrees=True)]
            points2 = [(x[0]+cx, x[1]+cy) for x in rotate_2d(points, angle+90+b, degrees=True)]
            draw.polygon(points1, fill=shape_color)
            draw.polygon(points2, fill=shape_color)
        elif shape == "trapezoid":
            h = int(r*np.random.randint(40, 50) / 100)
            w = np.sqrt(r*r - h*h)
            o = int(w*np.random.randint(40, 70) / 100)
            ltr_size = int(min(w, h)*np.random.randint(85, 95) / 100)
            points = [( +w, +h ),( +w-o, -h ),( -w+o, -h ),( -w, +h )]
            points = [(x[0]+cx, x[1]+cy) for x in rotate_2d(points, angle, degrees=True)]
            draw.polygon(points, fill=shape_color)
        elif shape == "star":
            sides = 5
            step = 2*np.pi / sides
            points = []
            c = r*np.random.randint(94, 100) / 100
            ratio = 1-np.sin(np.radians(36))*(np.tan(np.radians(18))+np.tan(np.radians(36)))
            for i in range(sides):
                points.append( (c*np.cos((step*i) + np.radians(-angle))) + cx )
                points.append( (c*np.sin((step*i) + np.radians(-angle))) + cy )
                points.append( (c*ratio*np.cos((step*i + step/2) + np.radians(-angle))) + cx )
                points.append( (c*ratio*np.sin((step*i + step/2) + np.radians(-angle))) + cy )
            draw.polygon(points, fill=shape_color)
            ltr_size = int(c*ratio*np.random.randint(90, 95) / 100)
        return (cx, cy), ltr_size, angle

    def draw_letter(self, draw, ltr_size, ltr_idx, ltr_color_idx):
        """ Do not use. this is called within draw_target.
            This function chooses a random font a draws the
            specified letter on a transparent PIL image. This
            image has the specified size, color, and angle."""
        font_path = "fonts/"+random.choice(os.listdir("fonts"))
        font = ImageFont.truetype(font_path, size=ltr_size*2)  # Double since ltr_size is based on radius
        letter_color = color_to_hsv(self.color_options[ltr_color_idx])
        w, h = draw.textsize(self.letter_options[ltr_idx], font=font)
        img = Image.new("RGBA", (w, h), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.text((0,0), self.letter_options[ltr_idx], fill=letter_color, font=font)
        img = img.rotate(np.random.uniform(0, 360), expand=1)
        return img

    def draw_target(self, img_size, min_size, fill_prob=1.0, transparent_bkg=False):
        """ Draws a random target on a transparent PIL image. Also returns the correct labels. """
        # Sample the label space with uniform random sampling
        bkg_color_idx, shp_color_idx, ltr_color_idx = np.random.choice(range(len(self.color_options)), 3, replace=False)
        alias_size = (int(self.alias_factor*img_size[0]), int(self.alias_factor*img_size[1]))
        has_target = np.random.rand() < fill_prob
        # Create a tranparent image. Transparent background allows PIL to overlay the image.
        target = Image.new('RGBA', size=alias_size, color=(0, 0, 0, 0))
        if has_target:
            draw = ImageDraw.Draw(target)
            shp_idx = np.random.randint(0, len(self.shape_options))
            ltr_idx = np.random.randint(0, len(self.letter_options))
            # Drawing puts the shape directly on the PIL image. Outputs are the center of the shape and the max letter size in pixels
            (cx, cy), ltr_size, angle = self.draw_shape(draw, img_size, min_size, shp_idx, shp_color_idx)
            letter = self.draw_letter(draw, ltr_size, ltr_idx, ltr_color_idx)
            ox, oy = letter.size
            temp = Image.new('RGBA', size=alias_size, color=(0, 0, 0, 0))
            temp.paste(letter, (cx-(ox//2), cy-(oy//2)), letter)  # Put letter with offset based on size
            target.alpha_composite(temp)  # removes the transparent aliasing border from the letter
        else:
            # Null labels for when there is no target
            angle, shp_idx, ltr_idx, shp_color_idx, ltr_color_idx = 0,0,0,0,0

        # If there are no backgrounds and no transparent_bkg arg, then use the bkg_color_idx.
        if self.backgrounds == None and not transparent_bkg:
            bkg_color = color_to_hsv(self.color_options[bkg_color_idx])
            img = Image.new('RGBA', size=alias_size, color=bkg_color)
            if has_target:  # Add the target to to the background
                img.paste(target, None, target)  # Alpha channel is the mask
            target = img

        label = {
            "has_target": int(has_target),
            "angle" : angle,
            "shape": shp_idx,
            "letter": ltr_idx,
            "shape_color": shp_color_idx,
            "letter_color": ltr_color_idx,
        }
        return target, label

    def get_background(self):
        """ Works like a iterator that randomly samples the backgrounds array. """
        self.bkg_count += 1
        if self.bkg_count == len(self.backgrounds):
            np.random.shuffle(self.bkg_idxs)
            self.bkg_count = 0
        return self.backgrounds[self.bkg_idxs[self.bkg_count]] 

    def gen_classify(self, img_size=None, min_size=None, fill_prob=1.0):
        """ Generate a cropped target with it's classification label. """
        if img_size == None:  # Use the default image size
            img_size = self.img_size
        else:  # otherwise make sure the input is a tuple
            img_size = (img_size,img_size) if type(img_size)!=tuple else img_size
        if min_size == None:
            min_size = self.min_size
        target, label = self.draw_target(img_size, min_size, fill_prob)
        if self.backgrounds is not None:
            bkg = self.get_background()
            img = T.RandomResizedCrop((int(self.alias_factor*img_size[1]), int(self.alias_factor*img_size[0])), scale=(0.08, 1.0), ratio=(3./4., 4./3.))(bkg)
            img.paste(target, None, target)  # Alpha channel is the mask
        else:
            # If there are no backgrounds, draw_target puts a random color.
            img = target
        img = img.resize(img_size).convert("RGB")
        return img, label

    def gen_segment(self, img_size=None, min_size=None, fill_prob=1.0):
        """ Generate an image with target mask. """
        if img_size == None:  # Use the default image size
            img_size = self.img_size
        else:  # otherwise make sure the input is a tuple
            img_size = (img_size,img_size) if type(img_size)!=tuple else img_size
        if min_size == None:
            min_size = self.min_size
        # Pick random gridsize based on input and target_size.
        bkg_w, bkg_h = img_size
        scale_w, scale_h = bkg_w//min_size, bkg_h//min_size  # Smallest grid cells based on the smallest target
        max_num = min(scale_w, scale_h)
        num = np.random.randint(1, max_num+1) if max_num>1 else 1 # Divisions along the smallest dimension
        # Scale divisions two both axis
        if bkg_w > bkg_h:
            num_w = int(scale_w/scale_h * num)
            num_h = num
        else:
            num_h = int(scale_h/scale_w * num)
            num_w = num
        step_w, step_h = bkg_w//num_w, bkg_h//num_h  # Rectangle size for each target
        # This mask is first used to place all the targets, then converted into a binary image
        place_targets = Image.new('RGBA', size=(int(bkg_w*self.alias_factor), int(bkg_h*self.alias_factor)), color=(0, 0, 0, 0))
        # Mask to fill the grid randomly with targets
        target_mask = np.random.rand(num_w*num_h) < fill_prob
        max_size = int(min(step_w, step_h))  # Targest target that can fit in the rectangle.
        # Number of pixels the target can be moved from the center of the rectangle
        offset_x, offset_y = int(step_w-max_size), int(step_h-max_size)
        for i in range(len(target_mask)):
            y = i // num_w
            x = (i - y*num_w)
            if target_mask[i]:
                target, label = self.draw_target((step_w, step_h), min_size, transparent_bkg=True)
                ox = np.random.randint(0, offset_x+1) if offset_x > 0 else 0
                oy = np.random.randint(0, offset_y+1) if offset_y > 0 else 0
                place_targets.paste(target, (int((x*step_w+ox)*self.alias_factor), int((y*step_h+oy)*self.alias_factor)), target)  # Alpha channel is the mask
        mask = Image.new('RGBA', size=(int(bkg_w*self.alias_factor), int(bkg_h*self.alias_factor)), color=(0, 0, 0, 0))
        mask.alpha_composite(place_targets)  # Removes the transparent aliasing border from the mask
        if self.backgrounds is not None:
            bkg = self.get_background()
            img = T.RandomResizedCrop((int(self.alias_factor*img_size[1]), int(self.alias_factor*img_size[0])), scale=(0.08, 1.0), ratio=(3./4., 4./3.))(bkg)
            img.paste(mask, None, mask)  # Alpha channel is the mask
        else:
            img = mask.convert("RGB")  # Return the mask in rgb
        img = img.resize(img_size).convert("RGB")
        # Convert the transparency to the binary mask
        mask = Image.fromarray(np.asarray(mask)[:,:,3])
        mask = mask.resize(img_size).convert("RGB")
        return img, mask

class LiveClassifyDataset(Dataset):
    def __init__(self, length, img_size, min_size, alias_factor=1, target_transforms=None,
                fill_prob=1.0, backgrounds=None, transforms=None):
        """ Dataset that makes generator object and calls it in __getitem__ """
        self.length = length
        self.gen = TargetGenerator(img_size, min_size, alias_factor, target_transforms, backgrounds)
        self.transforms = transforms if transforms else T.ToTensor()
        self.fill_prob = fill_prob

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        img, y = self.gen.gen_classify(fill_prob=self.fill_prob)
        target = list(y.values())
        return self.transforms(img), torch.tensor(target, dtype=torch.float).squeeze()


def visualize_classify(gen):
    nrows, ncols = 8, 8
    rows = []
    for i in range(nrows):
        row = [gen.gen_classify(img_size=64, min_size=28, fill_prob=0.9)[0] for j in range(ncols)]
        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()
    im.save("images/visualize_classify.png")

def visualize_segment(gen):
    nrows, ncols = 3, 2
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            img, mask = gen.gen_segment(img_size=(128, 128), min_size=28, fill_prob=0.5)
            row.append(img)
            row.append(mask)
        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    print(grid_img.shape)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()
    im.save("images/visualize_segment.png")

def dataset_stats(dataset, num=1000):
    mean, var = 0.0, 0.0
    t0 = time.time()
    t1 = t0
    for i in range(num):
        img, _ = dataset[i]  # __getitem__ , img in shape [W, H, C]
        # [1, C, H, W], expand so that the mean function can run on dim=0
        img = np.expand_dims((np.array(img)), axis=0)
        mean += np.mean(img, axis=(0, 2, 3))
        var += np.var(img, axis=(0, 2, 3))  # you can add var, not std
        if (i+1) % 100 == 0:
            t2 = time.time()
            print(f"{i+1}/{num} measured. Total time={t2-t0:.2f}s. Images per second {100/(t2-t1):.2f}.")
            t1 = t2
    print("mean :", mean/num)
    print("var :", var/num)
    print("std :", np.sqrt(var/num))

def visualize_batch(dataloader):
    from torchvision.utils import make_grid
    for images, labels in dataloader:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([]); ax.set_yticks([])
        ax.imshow(make_grid((images.detach()[:64]), nrow=8).permute(1, 2, 0))
        break
    fig.savefig('images/classify_processed.png', bbox_inches='tight')
    plt.show()

def time_dataloader(dataset, batch_size=64, max_num_workers=8):
    print(" * Time Dataloader...")
    for i in range(max_num_workers+1):
        ram_before = psutil.virtual_memory()[3]
        train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=False,
            num_workers=i, drop_last=True, persistent_workers=(True if i >0 else False))
        max_ram = 0
        ts = time.time()
        [_ for _ in train_loader]
        t0 = time.time()
        for batch_idx, (data, target) in enumerate(train_loader):
            r = psutil.virtual_memory()[3]
            max_ram = max(max_ram, r)
        ram_usage = (max_ram - ram_before)*1e-9  # GB
        duration = time.time()-t0
        print(f"{duration:.2f} seconds with {i} workers. {duration/(batch_idx+1):.2f} seconds per batch. {ram_usage:.3f} GB ram.")

if __name__ == "__main__":

    img_size = 32  # pixels, (input_size, input_size) or (width, height)
    min_size = 26  # pixels
    alias_factor = 2  # generate higher resolution targets and downscale, improves aliasing effects
    target_transforms = T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation="bicubic")
    backgrounds = r'C:\Users\lukeasargen\projects\aerial_backgrounds'
    fill_prob = 0.9

    backgrounds = load_backgrounds(backgrounds)
    generator = TargetGenerator(img_size, min_size, alias_factor, target_transforms, backgrounds)
    # visualize_classify(generator)
    # visualize_segment(generator)

    batch_size = 64
    train_size = 1024
    shuffle = False
    num_workers = 0
    drop_last = True
    train_transforms = T.Compose([
        CustomTransformation(),
        T.ToTensor(),
        AddGaussianNoise(0.01),
    ])

    dataset = LiveClassifyDataset(train_size, img_size, min_size, alias_factor, target_transforms, fill_prob, backgrounds, train_transforms)
    # dataset_stats(dataset, num=1000)
    # time_dataloader(dataset, batch_size=256, max_num_workers=8)

    loader = DataLoader(dataset=dataset, batch_size=batch_size ,shuffle=shuffle,
        num_workers=num_workers, drop_last=drop_last, persistent_workers=(True if num_workers > 0 else False))
    # visualize_batch(loader)

