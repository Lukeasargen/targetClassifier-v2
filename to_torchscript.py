import os

import torch

from classify import ClassifyModel
from detect import SegmentModel


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    checkpoint_path = r"logs\classify\version_32\checkpoints"
    checkpoint_path = checkpoint_path+"/"+os.listdir(checkpoint_path)[0]
    print(checkpoint_path)
    model = ClassifyModel.load_from_checkpoint(checkpoint_path)
    script = model.to_torchscript()
    torch.jit.save(script, "dev/classify.pt")

    classify_model = torch.jit.load("dev/classify.pt", map_location=device)
    inp = torch.rand(4,3,48,48).to(device)
    with torch.no_grad():
        y = classify_model(inp)
    print(y)

    checkpoint_path = r"logs\segment\version_7\checkpoints"
    checkpoint_path = checkpoint_path+"/"+os.listdir(checkpoint_path)[0]
    print(checkpoint_path)
    model = SegmentModel.load_from_checkpoint(checkpoint_path)
    script = model.to_torchscript()
    torch.jit.save(script, "dev/detect.pt")

    segment_model = torch.jit.load("dev/detect.pt", map_location=device)
    inp = torch.rand(4,3,2592,1944).to(device)
    with torch.no_grad():
        y = segment_model(inp)
    print(y.shape)


if __name__ == "__main__":
    main()