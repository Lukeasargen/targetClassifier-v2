import os
import random
import time
import numpy as np

from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T

from util import pil_loader

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

def load_backgrounds(folder_path):
    print(" * Loading backgrounds...")
    ts = time.time()
    backgrounds = [pil_loader(os.path.join(os.getcwd(), x)) for
                    x in os.scandir(folder_path) if not x.is_dir()]
    print(f" ** Backgrounds loaded. {time.time()-ts:.03f} seconds for {len(backgrounds)} images.")
    return backgrounds

class TargetGenerator():
    def __init__(self, img_size, min_size, alias_factor=1, target_transforms=None, backgrounds=None):
        self.img_size = (img_size,img_size) if type(img_size)!=tuple else img_size
        self.min_size = min_size
        self.alias_factor = alias_factor
        self.alias_size = tuple(i*alias_factor for i in self.img_size)
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
            "angle": 2,
            "shape": len(self.shape_options),
            "letter": len(self.letter_options),
            "shape_color": len(self.color_options),
            "letter_color": len(self.color_options),
        }
        self.num_outputs = sum(self.output_sizes.values())

    def draw_shape(self, draw, shp_idx, shp_color_idx):
        """ Do not use directly. This is called within draw_target.
            This function draws the specified shape and color.
            Scale and rotation are uniformly sampled.
            It returns values that specify how to draw the letter.
        """
        shape_color = color_to_hsv(self.color_options[shp_color_idx])
        shape = self.shape_options[shp_idx]
        # Uniformly sample that target size.
        # Half this is the radius of the circumscribed circle. Polygon vertices are on this circle.
        r = (np.random.uniform(self.min_size, min(self.alias_size))) // 2
        cx, cy = np.random.randint(r, self.alias_size[0]-r), np.random.randint(r, self.alias_size[1]-r)
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
            ltr_size = int(c*ratio*np.random.randint(80, 95) / 100)

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

    def draw_target(self):
        """ Draws a random target on a transparent PIL image. Also returns the correct labels. """
        # Sample the label space with uniform random sampling
        bkg_color_idx, shp_color_idx, ltr_color_idx = np.random.choice(range(len(self.color_options)), 3, replace=False)
        shp_idx = np.random.randint(0, len(self.shape_options))
        ltr_idx = np.random.randint(0, len(self.letter_options))
        # Create a tranparent image. Transparent background allows PIL to overlay the image.
        target = Image.new('RGBA', size=(self.alias_size[0], self.alias_size[1]), color=(100, 0, 0, 0))
        draw = ImageDraw.Draw(target)
        # Drawing puts the shape directly on the PIL image
        # outputs are the center of the shape and the max letter size in pixels
        (cx, cy), ltr_size, angle = self.draw_shape(draw, shp_idx, shp_color_idx)
        letter = self.draw_letter(draw, ltr_size, ltr_idx, ltr_color_idx)
        ox, oy = letter.size
        temp = Image.new('RGBA', size=self.alias_size, color=(0, 0, 0, 0))
        temp.paste(letter, (cx-(ox//2), cy-(oy//36)-(oy//2)), letter)  # Put letter with offset based on size
        target.alpha_composite(temp)  # removes the transparent aliasing border from the letter

        # If there are no backgrounds, then use the bkg_color_idx.
        if self.backgrounds == None:
            bkg_color = color_to_hsv(self.color_options[bkg_color_idx])
            img = Image.new('RGB', size=self.alias_size, color=bkg_color)
            img.paste(target, None, target)  # Alpha channel is the mask
            target = img

        label = {
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

    def gen_classify(self):
        """ Generate a cropped target with it's classification label. """
        target, label = self.draw_target()
        if self.backgrounds is not None:
            bkg = self.get_background()
            img = T.RandomResizedCrop((self.alias_size[1], self.alias_size[0]), scale=(0.08, 1.0), ratio=(3./4., 4./3.))(bkg)
            img.paste(target, None, target)  # Alpha channel is the mask
        else:
            # If there are no backgrounds, draw_target puts a random color.
            img = target
        img = img.resize(self.img_size).convert("RGB")
        return img, label

    def gen_segment(self):
        """ Generate an image with target mask. """
        img = None
        mask = None
        return img, mask

def visualize_classify(gen):
    nrows, ncols = 8, 8
    rows = []
    for i in range(nrows):
        row = [gen.gen_classify()[0] for j in range(ncols)]
        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()
    # im.save("images/visualize_classify.png")

def visualize_segment(gen):
    nrows, ncols = 5, 3
    rows = []
    for i in range(nrows):
        row = []
        for j in range(ncols):
            img, mask = gen.gen_segment(input_size=(400, 400), target_size=20, fill_prob=0.5)
            row.append(img)
            row.append(mask)
        rows.append( np.hstack(row) )
    grid_img = np.vstack(rows)
    print(grid_img.shape)
    im = Image.fromarray(grid_img.astype('uint8'), 'RGB')
    im.show()
    # im.save("images/visualize_segment.png")


if __name__ == "__main__":

    img_size = 32, 32  # pixels, (input_size, input_size) or (width, height)
    min_size = 26  # pixels
    alias_factor = 2  # generate higher resolution targets and downscale, improves aliasing effects
    backgrounds = None  # load_backgrounds('images/backgrounds')  # pil array
    target_transforms = T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation="bicubic")

    generator = TargetGenerator(img_size, min_size, alias_factor, target_transforms, backgrounds)

    # visualize_classify(generator)
    # visualize_segment(generator)

