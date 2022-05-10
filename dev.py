from numpy import blackman
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, precision_recall

output_sizes = {
    "has_target": 1,
    "angle": 2,
    "shape": 13,
    "letter": 34,
    "shape_color": 10,
    "letter_color": 10,
}
num_outputs = sum(output_sizes.values())

batch = 4

logits = {}
for k, v in output_sizes.items():
    logits[k] = torch.randn(batch, v)
target = torch.randint(low=0, high=10, size=(batch, len(output_sizes)), dtype=torch.float32)
target[:, 0] = target[:, 0]>1
target[:, 1] = torch.rand(batch)*360  # torch.tensor([0,90,180,270])

print(f"{target=}")


def predictions(logits):
    preds = {}
    for k in output_sizes.keys():
        if k=="has_target":
            preds["has_target"] = torch.sigmoid(logits["has_target"]).squeeze()
        elif k=="angle":
            preds["angle"] = F.normalize(torch.tanh(logits["angle"]), dim=1)
        else:
            preds[k] = torch.argmax(logits[k], dim=1)  # argmax of logits or softmax is the same index
    return preds

preds = predictions(logits)
print(f"{preds=}")

def custom_loss(logits, target):
    losses = {}
    mask = target[:,0]>0  # Only do the loss of the ones with targets
    for idx, k in enumerate(output_sizes.keys()):
        if k=="has_target":
            losses[k] = nn.BCEWithLogitsLoss()(logits[k].squeeze(), target[:,0])
        elif k=="angle":
            phasor = F.normalize(torch.tanh(logits["angle"]), dim=1)
            x = torch.sin(torch.deg2rad(target[:,1]))
            y = torch.cos(torch.deg2rad(target[:,1]))
            phasor_target = torch.stack([x, y]).permute(1,0)          
            losses[k] = nn.MSELoss()(phasor[mask], phasor_target[mask]) if sum(mask)>0 else 0
        else:
            losses[k] = nn.CrossEntropyLoss()(logits[k][mask], target[:, idx][mask].long()) if sum(mask)>0 else 0
    return losses

losses = custom_loss(logits, target)
print(f"{losses=}")

def calc_metrics(preds, target):
    metrics = {}
    mask = target[:,0]>0  # Only do the loss of the ones with targets
    print(f"{mask=}")
    for idx, (k,classes) in enumerate(output_sizes.items()):
        if k=="angle":
            x, y = preds[k][:,0], preds[k][:,1]
            angles = torch.rad2deg(torch.atan2(x, y))
            error = abs((target[:,1]-angles+900)%360-180)
            metrics[f"{k}/error"] = torch.mean(error[mask]) if sum(mask)>0 else 0
        else:
            if k=="has_target": preds[k] = preds[k]>0.5
            p = preds[k][mask]
            t = target[:, idx][mask].long()
            metrics[f"{k}/acc"] = accuracy(p, t) if sum(mask)>0 else 0
            if classes>2:
                precision, recall = precision_recall(p, t, num_classes=classes, average="macro", mdmc_average="global") if sum(mask)>0 else (0,0)
                metrics[f"{k}/precision"] = precision
                metrics[f"{k}/recall"] = recall
    return metrics

metrics = calc_metrics(preds, target)
print(f"{metrics=}")
