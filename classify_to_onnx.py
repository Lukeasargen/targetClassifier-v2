import matplotlib.pyplot as plt
import numpy as np
import torch
import onnxruntime as ort # pip install onnxruntime-gpu

from classify import ClassifyModel
from generator import TargetGenerator
from util import load_backgrounds

def main():
    checkpoint_path = r"logs\classify\version_23\checkpoints\epoch=169-step=17000.ckpt"
    model = ClassifyModel.load_from_checkpoint(checkpoint_path)

    filepath = "classify.onnx"
    input_sample = torch.randn((1, 3, 32, 32))
    model.to_onnx(filepath, input_sample, export_params=True)

    img_size = 32  # pixels, (input_size, input_size) or (width, height)
    min_size = 28  # pixels
    alias_factor = 3  # generate higher resolution targets and downscale, improves aliasing effects
    backgrounds = r'C:\Users\lukeasargen\projects\aerial_validate'
    fill_prob = 0.9
    backgrounds = load_backgrounds(backgrounds)
    generator = TargetGenerator(img_size, min_size, alias_factor, backgrounds)

    ort_session = ort.InferenceSession(filepath, providers=['CPUExecutionProvider'])

    rows = 3
    cols = 4
    scale = 3
    fig = plt.figure(figsize=(scale*cols, scale*rows))
    for i in range(1, cols*rows+1):
        img, label = generator.gen_classify(fill_prob=fill_prob)
        img_input = np.expand_dims(np.array(img).astype(np.float32).transpose((2, 0, 1)), axis=0)/255
        ort_inputs = {ort_session.get_inputs()[0].name: img_input}
        preds = ort_session.run(None, ort_inputs)

        ax = fig.add_subplot(rows, cols, i)
        ax.imshow(img)
        title_str = f"Has Target {100*preds[0]:.2f}%"
        title_color = "g" if (preds[0]>0.75)==label['has_target'] else "r"
        if label['has_target']:
            x, y = preds[1][:,0], preds[1][:,1]
            angle = float(np.degrees(np.arctan2(x, y)))
            angle = (angle+360)%360
            title_str += f" ({float(label['angle']):.0f}° / {angle:.0f}°)"
        ax.set_title(title_str, color=title_color)

        if label['has_target']:
            y_str = f"{generator.color_options[label['shape_color']]} {generator.shape_options[label['shape']]} / "
            y_str += f"{generator.color_options[int(preds[4])]} {generator.shape_options[int(preds[2])]}"
            y_color = "g" if (label['shape_color']==int(preds[4])) and (label['shape']==int(preds[2])) else "r"
            ax.set_ylabel(y_str, color=y_color)
            x_str = f"{generator.color_options[label['letter_color']]} {generator.letter_options[label['letter']]} / "
            x_str += f"{generator.color_options[int(preds[5])]} {generator.letter_options[int(preds[3])]}"
            x_color = "g" if (label['letter_color']==int(preds[5])) and (label['letter']==int(preds[3])) else "r"
            ax.set_xlabel(x_str, color=x_color)

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()