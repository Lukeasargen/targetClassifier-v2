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

def slice_output(x):
    """ Slice the outputs for task. """
    out = {}
    c = 0
    for k,v in output_sizes.items():
        out[k] = x[..., c:c+v]
        c += v
    return out


batch = 4
prediction_heads = 2
num_outputs = 70

x = torch.randn(batch, prediction_heads, num_outputs)
target = torch.randint(low=0, high=10, size=(batch, len(output_sizes)), dtype=torch.float32)
target[:, 0] = target[:, 0]>1
target[:, 1] = torch.rand(batch)*360
# print(f"{target=}")

logits = slice_output(x)

# prediction function
# preds = {}
# preds["has_target"] = torch.mean(torch.sigmoid(logits["has_target"]), dim=1)>0.5
# phasor = torch.mean(torch.tanh(logits["angle"]), dim=1)
# preds["angle"] = torch.rad2deg(torch.atan2(phasor[:,1], phasor[:,0]))
# for idx, k in enumerate(["shape", "letter", "shape_color", "letter_color"]):
#     v = logits[k]
#     soft = torch.mean(torch.softmax(v, dim=2), dim=1)
#     preds[k] = torch.argmax(soft, dim=1)

# for idx, (k, p) in enumerate(preds.items()):
#     if k=="has_target":
#         t = target[:,0].long()
#         acc = accuracy(p, t)
#         print(f"{k} {acc=}")
#     elif k=="angle":
#         t = target[:,1]
#         error = torch.mean(abs((t-p+900)%360-180))
#         print(k, error)
#     else:
#         t = target[:, idx].long()
#         acc = accuracy(p, t)
#         print(f"{k} {acc=}")





# pred = torch.argmax(logits, dim=1)
# acc = accuracy(pred, target)
# avg_precision, avg_recall = precision_recall(pred, target, num_classes=self.hparams.num_classes,
#                                                 average="macro", mdmc_average="global")
# weighted_f1 = f1_score(pred, target, num_classes=self.hparams.num_classes,
#                     threshold=0.5, average="weighted")


# print("target", target)

# training loss function
losses = {}
bl = logits["has_target"].reshape(logits["has_target"].shape[0], prediction_heads)
bt = target[:,0].unsqueeze(dim=1).repeat(1, prediction_heads)
losses["has_target"] = nn.BCEWithLogitsLoss(reduction='none')(bl, bt)

mask = target[:,0]>0 # Only do the loss of the ones with targets
print(mask)

al = F.normalize(torch.tanh(logits["angle"]), dim=2)
cs = [torch.cos(torch.deg2rad(target[:,1])), torch.sin(torch.deg2rad(target[:,1]))]
at = torch.stack(cs, dim=0).permute(1,0).repeat(1, 1, prediction_heads)
at = at.reshape(al.shape[0], prediction_heads, 2)
aa = torch.sum(nn.MSELoss(reduction='none')(al, at), dim=2)
losses["angle"] = torch.sum(nn.MSELoss(reduction='none')(al, at), dim=2)[mask,...]


for idx, k in enumerate(["shape", "letter", "shape_color", "letter_color"]):
    t = target[:, idx+2].repeat(prediction_heads).reshape(logits[k].shape[0], prediction_heads)
    l = logits[k].reshape(logits[k].shape[0], -1, prediction_heads)
    losses[k] = nn.CrossEntropyLoss(reduction='none')(l, t.long())[mask,...]

# print( losses )
