import cv2
import numpy as np
import torch
import torchvision.transforms as T

from classify import ClassifyModel
from detect import SegmentModel
from generator import TargetGenerator
from util import load_backgrounds, quantize_to_int
from util import color_options, shape_options, letter_options

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def detect_pipline(model, pil_image, thresh=0.5, expand_ratio=1, min_size=0):
    w, h = pil_image.size
    cropped = crop_center(pil_image, quantize_to_int(w,16), quantize_to_int(h,16))
    img_ten = T.ToTensor()(cropped).unsqueeze(0)
    with torch.no_grad():
        preds = model(img_ten)
    preds = preds[0,0].numpy()
    # preds = cv2.erode(preds, None, iterations=1)
    preds = (255*(preds>thresh)).astype(np.uint8)
    contours, hierarchy  = cv2.findContours(preds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    patches = []
    for (i, c) in enumerate(contours):
        bbox = cv2.boundingRect(c)
        (x, y, w, h) = bbox
        if min(w,h)>min_size:
            cx, cy = x+w//2, y+h//2
            r = int(expand_ratio*max(h,w))
            tx, ty, bx, by = cx-r//2, cy-r//2, cx+r//2, cy+r//2
            # Corpping pads black pixels at the edge of the image
            patch = pil_image.crop((tx, ty, bx, by))
            patches.append(patch)
    return patches

def classify_pipline(model, image, img_size):
    resized = image.resize((img_size,img_size))
    resized = T.GaussianBlur(3,1)(resized)
    img_ten = T.ToTensor()(resized).unsqueeze(0)
    with torch.no_grad():
        preds = model(img_ten)
    label = {}
    label["has_target"] = float(preds["has_target"])
    label["angle"] = (float(np.degrees(np.arctan2(preds["angle"][:,0], preds["angle"][:,1])))+360)%360
    label["shape_idx"] = int(preds["shape"])
    label["shape"] = shape_options[int(preds["shape"])]
    label["letter_idx"] = int(preds["letter"])
    label["letter"] = letter_options[int(preds["letter"])]
    label["shape_color_idx"] = int(preds["shape_color"])
    label["shape_color"] = color_options[int(preds["shape_color"])]
    label["letter_color_idx"] = int(preds["letter_color"])
    label["letter_color"] = color_options[int(preds["letter_color"])]
    return label

def main():
    segment_path = r"logs\segment\version_7\checkpoints\epoch=34-step=3500.ckpt"
    segment_model = SegmentModel.load_from_checkpoint(segment_path).eval()

    classify_path = r"logs\classify\version_23\checkpoints\epoch=169-step=17000.ckpt"
    classify_model = ClassifyModel.load_from_checkpoint(classify_path).eval()

    # 2592×1944 4056x3040
    img_size = (1024,840)  # pixels, (input_size, input_size) or (width, height)
    min_size = 28  # pixels
    alias_factor = 3  # generate higher resolution targets and downscale, improves aliasing effects
    backgrounds = r'C:\Users\lukeasargen\projects\aerial_validate'
    fill_prob = 0.1
    backgrounds = load_backgrounds(backgrounds)
    generator = TargetGenerator(img_size, min_size, alias_factor, backgrounds)

    img_color, mask = generator.gen_segment(fill_prob=fill_prob)
    patches = detect_pipline(segment_model, img_color, expand_ratio=1.5, min_size=10)
    labels = [classify_pipline(classify_model, patch, img_size=32) for patch in patches]


    import os
    import random
    import string
    import matplotlib.pyplot as plt
    name = ''.join(random.choice(string.ascii_lowercase) for i in range(4))
    print(f"{len(patches)=}")
    print(f"{name=}")
    out_folder = "images/pipline/"+name+"/"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)
    img_color.save(out_folder+name+".jpg")
    for idx, (patch, label) in enumerate(zip(patches, labels)):
        scale = 6
        fig, ax = plt.subplots(figsize=(scale, scale))
        ax.imshow(patch)
        title_str = f"""Target {100*label["has_target"]:.2f}%
        {label["angle"]:.0f}°
        {label["shape_color"]} {label["shape"]}
        {label["letter_color"]} {label["letter"]}"""
        ax.set_title(title_str)
        fig.tight_layout()
        out_path = os.path.join(out_folder, f"{name}_{idx}.jpg")
        fig.savefig(out_path, bbox_inches='tight')
        plt.close(fig)





if __name__ == "__main__":
    main()