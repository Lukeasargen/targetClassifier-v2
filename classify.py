import os
import argparse

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy
from torchvision.transforms import Normalize
import torchvision.transforms as T

from generator import load_backgrounds, LiveClassifyDataset
from util import quantize_to_int
from util import AddGaussianNoise, CustomTransformation

def get_args():
    parser = argparse.ArgumentParser()
    # Init and setup
    parser.add_argument('--seed', type=int)
    parser.add_argument('--name', default='classify', type=str)
    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--ngpu', default=1, type=int)
    parser.add_argument('--benchmark', default=False, action='store_true')
    parser.add_argument('--precision', default=32, type=int, choices=[16, 32])
    # Dataset
    parser.add_argument('--train_size', default=320, type=int)
    parser.add_argument('--img_size', default=32, type=int)
    parser.add_argument('--min_size', default=20, type=int)
    parser.add_argument('--alias_factor', default=2.0, type=float)
    parser.add_argument('--backgrounds', default=None, type=str)
    parser.add_argument('--fill_prob', default=0.5, type=float)
    parser.add_argument('--batch', default=16, type=int)
    parser.add_argument('--mean', nargs=3, default=[0.485, 0.456, 0.406], type=float)
    parser.add_argument('--std', nargs=3, default=[0.229, 0.224, 0.225], type=float)
    # Model parameters
    parser.add_argument('--feature_dim', default=64, type=int)
    parser.add_argument('--prediction_heads', default=1, type=int)
    parser.add_argument('--dropout', default=0.0, type=float)
    # Optimizer
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--opt', default='adam', type=str, choices=['sgd', 'adam', 'adamw'])
    parser.add_argument('--lr', default=4e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--nesterov', default=False, action='store_true')
    parser.add_argument('--weight_decay', default=0.0, type=float)
    parser.add_argument('--accumulate', default=1, type=int)
    # Scheduler
    parser.add_argument('--scheduler', default=None, type=str, choices=['step', 'exp', 'plateau'])
    parser.add_argument('--lr_gamma', default=0.2, type=float)
    parser.add_argument('--milestones', nargs='+', default=[10, 15], type=int)
    parser.add_argument('--plateau_patience', default=20, type=int)
    args = parser.parse_args()
    return args

class ClassifyModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super(ClassifyModel, self).__init__()
        self.save_hyperparameters()
        channels = [
            quantize_to_int(self.hparams.feature_dim/8,8),
            quantize_to_int(self.hparams.feature_dim/4,8),
            quantize_to_int(self.hparams.feature_dim/2,8),
            quantize_to_int(self.hparams.feature_dim/1,8)
        ]
        self.features =nn.Sequential(
            Normalize(args.mean, args.std, inplace=True),
            nn.Conv2d(in_channels=3, out_channels=channels[0], kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[2]),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[3]),
            nn.AdaptiveAvgPool2d((1,1))
        )
        self.dropout = nn.Dropout(self.hparams.dropout)
        self.fc = nn.Conv2d(quantize_to_int(self.hparams.feature_dim,8), self.hparams.num_outputs*self.hparams.prediction_heads, kernel_size=1, bias=True)
        self.scheduler = None

    def forward(self, x):
        """ Used for inference. """
        x = self.features(x)
        x = self.fc(x)
        x = x.reshape(x.shape[0], self.hparams.prediction_heads, self.hparams.num_outputs)
        logits = self.slice_output(x)
        preds = self.predictions(logits)
        return preds

    def slice_output(self, x):
        """ Slice the outputs for task. """
        out = {}  # logits dim: batch, heads, outputs
        c = 0
        for k,v in self.hparams.output_sizes.items():
            out[k] = x[..., c:c+v]
            c += v
        return out

    def predictions(self, logits):
        preds = {}
        preds["has_target"] = torch.mean(torch.sigmoid(logits["has_target"]), dim=1)>0.5
        phasor = torch.mean(torch.tanh(logits["angle"]), dim=1)
        preds["angle"] = torch.rad2deg(torch.atan2(phasor[:,1], phasor[:,0]))
        for idx, k in enumerate(["shape", "letter", "shape_color", "letter_color"]):
            v = logits[k]
            soft = torch.mean(torch.softmax(v, dim=2), dim=1)
            preds[k] = torch.argmax(soft, dim=1)
        return preds

    def custom_loss(self, logits, target):
        """ Compute the loss from the logits() dictionary. 
            Each task has a loss shape [batch, heads].
        """
        losses = {}
        bl = logits["has_target"].reshape(logits["has_target"].shape[0], self.hparams.prediction_heads)
        bt = target[:,0].unsqueeze(dim=1).repeat(1, self.hparams.prediction_heads)
        losses["has_target"] = nn.BCEWithLogitsLoss(reduction='none')(bl, bt)
        al = F.normalize(torch.tanh(logits["angle"]), dim=2)
        cs = [torch.cos(torch.deg2rad(target[:,1])), torch.sin(torch.deg2rad(target[:,1]))]
        at = torch.stack(cs,dim=0).permute(1,0).repeat(1, 1, self.hparams.prediction_heads)
        at = at.reshape(logits["angle"].shape[0], self.hparams.prediction_heads, 2)
        losses["angle"] = torch.sum(nn.MSELoss(reduction='none')(al, at), dim=2)
        for idx, k in enumerate(["shape", "letter", "shape_color", "letter_color"]):
            t = target[:, idx+2].repeat(self.hparams.prediction_heads).reshape(logits[k].shape[0], self.hparams.prediction_heads)
            l = logits[k].reshape(logits[k].shape[0], -1, self.hparams.prediction_heads)
            losses[k] = nn.CrossEntropyLoss(reduction='none')(l, t.long())
        return losses

    def batch_step(self, batch):
        data, target = batch
        x = self.features(data)
        x = self.fc(x)
        x = x.reshape(x.shape[0], self.hparams.prediction_heads, self.hparams.num_outputs)
        logits = self.slice_output(x)
        preds = self.predictions(logits)
        losses = self.custom_loss(logits, target)
        metrics = {"loss": 0}
        for idx, (k,v) in enumerate(losses.items()):
            avg = torch.mean(v)  # Task average loss
            # TODO: detach loss for tasks that are done training
            metrics["loss"] += avg  # use this as the total training loss
            metrics[f"{k}/loss"] = avg  # save loss each step
            if k=="angle":
                t = target[:,1]
                metrics[f"{k}/error"] = torch.mean(abs((t-preds[k]+900)%360-180))
            else:
                t = target[:, idx].long()
                metrics[f"{k}/acc"] = accuracy(preds[k], t)
                # TODO : add precision, recall, f1
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.batch_step(batch)
        for k, v in metrics.items():
            key = "{}".format(k)
            self.log(key, v, on_step=True, on_epoch=True)
        return metrics
    
    def training_epoch_end(self, outputs):
        # TODO: check which tasks to stop training
        # Log lr
        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.add_scalar('Learning Rate', lr, global_step=self.current_epoch)
        # Step scheduler
        if self.scheduler:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(avg_loss)
            elif type(self.scheduler) in [torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.ExponentialLR]:
                self.scheduler.step()

    def configure_optimizers(self):
        """https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3"""
        def add_weight_decay(module, weight_decay, lr):
            decay = []
            no_decay = []
            for name, param in module.named_parameters():
                if param.requires_grad:
                    if len(param.shape) == 1:  # Bias and bn parameters
                        no_decay.append(param)
                    else:
                        decay.append(param)
            return [{'params': no_decay, 'lr': lr,  'weight_decay': 0.0},
                    {'params': decay, 'lr': lr, 'weight_decay': weight_decay}]

        if self.hparams.weight_decay != 0:
            params = add_weight_decay(self, self.hparams.weight_decay, self.hparams.lr)
        else:
            params = self.parameters()

        if self.hparams.opt == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr, momentum=self.hparams.momentum,
                            nesterov=self.hparams.nesterov)
        elif self.hparams.opt == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr)
        elif self.hparams.opt == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.hparams.lr)

        if self.hparams.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.lr_gamma)
        elif self.hparams.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.lr_gamma, patience=self.hparams.plateau_patience, verbose=False)
        elif self.hparams.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_gamma)

        return optimizer

if __name__ == "__main__":
    args = get_args()

    backgrounds = load_backgrounds(args.backgrounds) if args.backgrounds is not None else args.backgrounds
    target_transforms = T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation="bicubic")
    train_transforms = T.Compose([
        CustomTransformation(),
        T.ToTensor(),
        AddGaussianNoise(0.01),
    ])

    print(" * Creating datasets and dataloaders...")
    train_dataset = LiveClassifyDataset(args.train_size, args.img_size, args.min_size, args.alias_factor,
            target_transforms, args.fill_prob, backgrounds, train_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch ,shuffle=False,
        num_workers=args.workers, drop_last=True, persistent_workers=(True if args.workers > 0 else False))
    
    args.color_options = train_dataset.gen.color_options
    args.shape_options = train_dataset.gen.shape_options
    args.letter_options = train_dataset.gen.letter_options
    args.output_sizes = train_dataset.gen.output_sizes
    args.num_outputs = train_dataset.gen.num_outputs

    model = ClassifyModel(**vars(args))

    pl.seed_everything(args.seed)

    # Increment to find the next availble name
    logger = TensorBoardLogger(save_dir="logs", name=args.name)    
    dirpath = f"logs/{args.name}/version_{logger.version}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    # callbacks = [
    #     ModelCheckpoint(
    #         monitor=args.save_monitor+"/val_epoch",
    #         dirpath=dirpath,
    #         filename="topk/{epoch:d}-{step}-{accuracy/val_epoch:.4f}",
    #         save_top_k=args.save_top_k,
    #         mode='min' if args.save_monitor=='loss' else 'max',
    #         period=1,  # Check every validation epoch
    #         save_last=True,
    #         save_on_train_epoch_end=False,
    #     )
    # ]

    trainer = pl.Trainer(
        accumulate_grad_batches=args.accumulate,
        benchmark=args.benchmark,  # cudnn.benchmark
        # callbacks=callbacks,
        # check_val_every_n_epoch=args.val_interval,
        deterministic=True,  # cudnn.deterministic
        gpus=args.ngpu,
        logger=logger,
        precision=args.precision,
        progress_bar_refresh_rate=1,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        # limit_val_batches=args.val_percent,
        log_every_n_steps=1,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
    )
