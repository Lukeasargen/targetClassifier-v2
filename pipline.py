import cv2
import numpy as np
import torch
import torchvision.transforms as T

from classify import ClassifyModel
from detect import SegmentModel
from util import quantize_to_int, pil_loader
from util import color_options, shape_options, letter_options

def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))

def detect_pipline(model, pil_image, thresh=0.5, expand_ratio=1, min_size=0):
    # TODO : patch image and submit as a batch to the gpu
    # TODO : get meta data, location, size
    w, h = pil_image.size
    cropped = crop_center(pil_image, quantize_to_int(w,16), quantize_to_int(h,16))
    img_ten = T.ToTensor()(cropped).unsqueeze(0)
    with torch.no_grad():
        preds = model(img_ten.to(model.device)).cpu()
    preds = preds[0,0].numpy()  # Get the first channel from the first sample
    torch.cuda.empty_cache()
    # preds = cv2.erode(preds, None, iterations=1)
    # preds = cv2.dilate(preds, None, iterations=1)
    preds = (255*(preds>thresh)).astype(np.uint8)
    contours, hierarchy  = cv2.findContours(preds, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    patches = []
    for (i, c) in enumerate(contours):
        meta = {}
        bbox = cv2.boundingRect(c)
        (x, y, w, h) = bbox
        if min(w,h)>min_size:
            cx, cy = x+w//2, y+h//2
            r = int(expand_ratio*max(h,w))
            tx, ty, bx, by = cx-r//2, cy-r//2, cx+r//2, cy+r//2
            # TODO : strictly crop within the full image
            # Cropping pads black pixels at the edge of the image
            crop = cropped.crop((tx, ty, bx, by))
            meta["thumbnail"] = crop
            meta["center_pixel"] = (cx,cy)
            meta["square_size"] = r
            patches.append(meta)
    return patches

def classify_pipline(model, patch, img_size=32):
    resized = patch["thumbnail"].resize((img_size,img_size))
    # resized = T.GaussianBlur(3,1)(resized)
    img_ten = T.ToTensor()(resized).unsqueeze(0)
    with torch.no_grad():
        preds = model(img_ten.to(model.device))
    patch["has_target"] = float(preds["has_target"])
    patch["has_letter"] = float(preds["has_letter"])
    patch["angle"] = float(preds["angle"])
    patch["shape_idx"] = int(preds["shape"])
    patch["shape"] = shape_options[int(preds["shape"])]
    patch["letter_idx"] = int(preds["letter"])
    patch["letter"] = letter_options[int(preds["letter"])]
    patch["shape_color_idx"] = int(preds["shape_color"])
    patch["shape_color"] = color_options[int(preds["shape_color"])]
    patch["letter_color_idx"] = int(preds["letter_color"])
    patch["letter_color"] = color_options[int(preds["letter_color"])]
    return patch

def test_pipline(folder_path, segment_model, classify_model):
    import time
    import os
    import matplotlib.pyplot as plt

    out_folder = "images/pipline/"
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    paths = [os.path.join(os.getcwd(), x) for x in os.scandir(folder_path)]
    images = [pil_loader(path) for path in paths]

    # from util import load_backgrounds
    # from generator import TargetGenerator
    # img_size = 256  # pixels, (input_size, input_size) or (width, height)
    # min_size = 28  # pixels
    # alias_factor = 3  # generate higher resolution targets and downscale, improves aliasing effects
    # backgrounds = r'C:\Users\lukeasargen\projects\aerial_validate'
    # fill_prob = 0.5
    # backgrounds = load_backgrounds(backgrounds)
    # generator = TargetGenerator(img_size, min_size, alias_factor, backgrounds)

    # count = 4
    # import random
    # import string
    # paths = [''.join(random.choice(string.ascii_lowercase) for i in range(5)) for i in range(count)]
    # images = [generator.gen_segment(fill_prob=fill_prob)[0] for i in range(count)]

    for path, image in zip(paths, images):

        name, ext = os.path.splitext(os.path.basename(path))
        with torch.cuda.amp.autocast():
            t0 = time.time()
            patches = detect_pipline(segment_model, image, expand_ratio=1.4, min_size=20)
            t1 = time.time()
            patches = [classify_pipline(classify_model, patch, img_size=48) for patch in patches]
            t2 = time.time()
            print(f"{image.size} total={t2-t0:.4f}s. detect={t1-t0:.4f}s. classify={t2-t1:.4f}s. patches={len(patches)}. {name+ext}")

        # image.save(out_folder+name+".jpg")
        for idx, label in enumerate(patches):
            scale = 6
            fig, ax = plt.subplots(figsize=(scale, scale))
            ax.imshow(label["thumbnail"])
            title_str = f"""Has Target {100*label["has_target"]:.2f}%
            Has Letter {100*label["has_letter"]:.2f}%
            {label["shape_color"]} {label["shape"]}
            {label["letter_color"]} {label["letter"]} @ {label["angle"]:.0f}°"""
            title_color = "g" if ((label["has_target"]>0.9) and (label["has_letter"]>0.9)) else "r"
            ax.set_title(title_str, color=title_color)
            fig.tight_layout()
            out_path = os.path.join(out_folder, f"{name}_{idx}.jpg")
            fig.savefig(out_path, bbox_inches='tight')
            plt.close(fig)

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    import os
    segment_path = r"logs\segment\version_7\checkpoints"
    segment_path = segment_path+"/"+os.listdir(segment_path)[0]
    print(segment_path)
    segment_model = SegmentModel.load_from_checkpoint(segment_path, map_location=device).eval().to(device)

    classify_model = ClassifyModel.load_from_checkpoint("classify34.ckpt", map_location=device).eval().to(device)

    root = r'C:\Users\lukeasargen\projects\aerial_real/'  # image folder, aerial_backgrounds, aerial_cropped, aerial_real, aerial_validate
    test_pipline(root, segment_model, classify_model)

if __name__ == "__main__":
    main()