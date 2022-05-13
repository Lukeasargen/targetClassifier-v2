import os
import argparse
from collections import Counter

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, ExponentialLR
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy, precision_recall
import torchvision.transforms as T

from generator import LiveSegmentDataset
from models import UNet
from util import load_backgrounds
from util import AddGaussianNoise, CustomTransformation
from util import pixel_accuracy, jaccard_iou, dice_coeff, tversky_measure, focal_metric

def get_args():
    parser = argparse.ArgumentParser()
    # Init and setup
    parser.add_argument('--seed', default=42, type=int, help="int. default=42. deterministic seed. cudnn.deterministic is always set True by deafult.")
    parser.add_argument('--name', default='segment', type=str, help="str. default=classify. Tensorboard name and log folder name.")
    parser.add_argument('--workers', default=0, type=int, help="int. default=0. Dataloader num_workers. good practice is to use number of cpu cores.")
    parser.add_argument('--gpus', nargs="+", default=None, type=int, help="str. default=None (cpu). gpus to train on. see pl multi_gpu docs for details.")
    parser.add_argument('--benchmark', default=False, action='store_true', help="store_true. set cudnn.benchmark.")
    parser.add_argument('--precision', default=32, type=int, choices=[16, 32], help="int. default=32. 32 for full precision and 16 uses pytorch amp")
    # Dataset
    parser.add_argument('--img_size', default=128, type=int, help="int. default=32. input size in pixels.")
    parser.add_argument('--min_size', default=20, type=int, help="int. default=20. smallest target size in pixels.")
    parser.add_argument('--alias_factor', default=2.0, type=float, help="float. default=2.0. generate higher resolution images and downscale to help with aliasing.")
    parser.add_argument('--backgrounds', default=None, type=str, help="str. Path to folder with backgrounds images.")
    parser.add_argument('--fill_prob', default=0.5, type=float, help="float. percentage of images that have targets.")
    parser.add_argument('--mean', nargs=3, default=[0.485, 0.456, 0.406], type=float, help="3 floats. default is imagenet [0.485, 0.456, 0.406].")
    parser.add_argument('--std', nargs=3, default=[0.229, 0.224, 0.225], type=float, help="3 floats. default is imagenet [0.229, 0.224, 0.225].")
    # Model parameters
    parser.add_argument('--filters', nargs='+', default=[16, 16, 32, 64], type=int, help="floats. default is [16, 16, 32, 64]. number of filters in each block, first value is the output dim of the stem and the final value is the feature dim.")
    parser.add_argument('--act', default=None, type=str, choices=['gelu', 'leaky_relu', 'relu', 'relu6', 'sigmoid', 'silu', 'tanh'], help="str. default=None. activation. use gelu, leaky_relu, relu, relu6, sigmoid, silu, tanh")
    parser.add_argument('--downsample', default='avg', type=str, choices=['avg', 'max', 'blur'], help="str. default=avg. activation. use avg, max, blur")
    parser.add_argument('--loss', nargs='+', default='jaccard', type=str, choices=['bce', 'jaccard', 'dice', 'tversky', 'focal'], help="str. default=jaccard. activation. use bce, jaccard, dice, tversky, focal")
    # Training hyperparameter
    parser.add_argument('--train_size', default=320, type=int, help="int. default=320. number of images in 1 epoch.")
    parser.add_argument('--batch', default=16, type=int, help="int. default=1. batch size.")
    parser.add_argument('--epochs', default=10, type=int, help="int. deafult=10. number of epochs")
    parser.add_argument('--add_noise', default=0.01, type=float, help="float. default=0.01. gaussian noise on the input.")
    # Optimizer
    parser.add_argument('--opt', default='adam', type=str, choices=['sgd', 'adam', 'adamw'], help="str. default=adam. use sgd, adam, or adamw.")
    parser.add_argument('--lr', default=4e-3, type=float)
    parser.add_argument('--momentum', default=0.9, type=float, help="float. default=0.9. sgd momentum value.")
    parser.add_argument('--nesterov', default=False, action='store_true', help="store_true. sgd with nestrov acceleration.")
    parser.add_argument('--weight_decay', default=0.0, type=float, help="float. default=0.0. weight decay for sgd and adamw. 0=no weight decay.")
    parser.add_argument('--adam_b1', default=0.9, type=float)
    parser.add_argument('--adam_b2', default=0.999, type=float)
    parser.add_argument('--lr_warmup_steps', default=0, type=int, help="int. deafult=0. linearly increase learning rate for this number of steps.")
    parser.add_argument('--accumulate', default=1, type=int, help="int. default=1. number of gradient accumulation steps. simulate larger batches when >1.")
    parser.add_argument('--grad_clip', default='value', type=str, choices=['value', 'norm'], help="str. default=value. pl uses clip_grad_value_ and clip_grad_norm_ from nn.utils.")
    parser.add_argument('--clip_value', default=0, type=float, help="float. default=0 is no clipping.")
    # Scheduler
    parser.add_argument('--scheduler', default=None, type=str, choices=['step', 'plateau', 'exp', 'cosine', 'one_cycle'], help="str. default=None. use step, plateau, exp, or cosine schedulers.")
    parser.add_argument('--lr_gamma', default=0.1, type=float, help="float. default=0.1. gamma for schedulers that scale the learning rate.")
    parser.add_argument('--milestones', nargs='+', default=[10, 15], type=int, help="ints. step scheduler milestones.")
    parser.add_argument('--plateau_patience', default=20, type=int, help="int. plateau scheduler patience. monitoring the train loss.")
    parser.add_argument('--min_lr', default=0.0, type=float, help="float. default=0.0. minimum learning rate set by Plateau scheduler.")
    args = parser.parse_args()
    return args

class SegmentModel(pl.LightningModule):
    def __init__(self, **kwargs):
        super(SegmentModel, self).__init__()
        self.save_hyperparameters()
        self.logits = nn.Sequential(
            T.Normalize(self.hparams.mean, self.hparams.std, inplace=True),
            UNet(3, 1, self.hparams.filters, self.hparams.act, self.hparams.downsample)
        )
        self.scheduler = None  # Set in configure_optimizers()
        self.opt_init_lr = None  # Set in configure_optimizers()

    def forward(self, x):
        """ Used for inference. """
        logits = self.logits(x)
        preds = torch.sigmoid(logits)
        return preds

    def calc_metrics(self, logits, targets, preds):
        metrics = {
            "loss/error": 1.0-pixel_accuracy(preds, targets, threshold=0.5),
            "loss/bce": F.binary_cross_entropy_with_logits(logits, targets),
            "loss/jaccard": 1.0-jaccard_iou(preds, targets, smooth=1.0),
            "loss/dice": 1.0-dice_coeff(preds, targets, smooth=1.0),
            "loss/tversky": 1.0-tversky_measure(preds, targets, alpha=0.3, beta=0.7, smooth=1.0),
            "loss/focal": focal_metric(logits, targets, alpha=0.5, gamma=2.0)
        }
        return metrics
        
    def batch_step(self, batch):
        data, masks = batch
        logits = self.logits(data)
        preds = torch.sigmoid(logits)
        metrics = self.calc_metrics(logits, masks, preds)
        metrics['loss'] = sum([metrics[f'loss/{k}'] for k in self.hparams.loss])
        # Detatch everything except the loss
        for k, v in metrics.items():
            if k!='loss' and torch.is_tensor(v):
                v = v.detach()
            metrics[k] = v
        return metrics

    def training_step(self, batch, batch_idx):
        metrics = self.batch_step(batch)

        # Update tensorboard for each train step
        for k, v in metrics.items():
            self.log(k, v, on_step=True, on_epoch=True)

        # Update the lr during warmup
        """ Code from https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html
            Except I didn't override optimizer_step() bc that would break gradient accumulation.
        """
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            opt = self.optimizers()
            lr_scale = min(1, float(self.trainer.global_step+1)/self.hparams.lr_warmup_steps)
            for pg, init_lr in zip(opt.param_groups, self.opt_init_lr):
                pg['lr'] = lr_scale*init_lr

        return metrics
    
    def training_epoch_end(self, outputs):
        totals = dict(sum((Counter(dict(x)) for x in outputs), Counter()))
        averages = {k: v/len(outputs) for k,v in totals.items()}

        # Log lr
        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.add_scalar('Learning Rate', lr, global_step=self.current_epoch)

        # Step these schedulers every epoch
        if type(self.scheduler) in [MultiStepLR, ExponentialLR]:
            self.scheduler.step()
        elif self.trainer.global_step >= self.hparams.lr_warmup_steps:
            if type(self.scheduler) in [ReduceLROnPlateau]:
                self.scheduler.step(averages["loss"])

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
            optimizer = torch.optim.SGD(params, lr=self.hparams.lr, momentum=self.hparams.momentum,nesterov=self.hparams.nesterov)
        elif self.hparams.opt == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.hparams.lr, betas=(self.hparams.adam_b1, self.hparams.adam_b2))
        elif self.hparams.opt == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.hparams.lr, betas=(self.hparams.adam_b1, self.hparams.adam_b2))

        # Keep a copy of the initial lr for each group because this will get overwritten during warmup steps
        self.opt_init_lr = [pg['lr'] for pg in optimizer.param_groups]

        if self.hparams.scheduler == 'step':
            self.scheduler = MultiStepLR(
                optimizer,
                milestones=self.hparams.milestones,
                gamma=self.hparams.lr_gamma)
        elif self.hparams.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.hparams.lr_gamma,
                patience=self.hparams.plateau_patience,
                min_lr=self.hparams.min_lr)
        elif self.hparams.scheduler == 'exp':
            self.scheduler = ExponentialLR(
                optimizer,
                gamma=self.hparams.lr_gamma)

        return optimizer

if __name__ == "__main__":
    args = get_args()

    backgrounds = load_backgrounds(args.backgrounds)
    target_transforms = T.RandomPerspective(distortion_scale=0.5, p=1.0, interpolation="bicubic")
    train_transforms = T.Compose([
        CustomTransformation(),
        T.ToTensor(),
        AddGaussianNoise(args.add_noise),
    ])

    print(" * Creating datasets and dataloaders...")
    train_dataset = LiveSegmentDataset(args.train_size, args.img_size, args.min_size, args.alias_factor,
                        target_transforms, args.fill_prob, backgrounds, train_transforms)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch ,shuffle=False,
        num_workers=args.workers, drop_last=True, persistent_workers=(True if args.workers > 0 else False))
    
    model = SegmentModel(**vars(args))

    pl.seed_everything(args.seed)

    # Increment to find the next availble name
    logger = TensorBoardLogger(save_dir="logs", name=args.name)    
    dirpath = f"logs/{args.name}/version_{logger.version}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    callbacks = [
        TQDMProgressBar(refresh_rate=1),
        # ModelCheckpoint(
        #     monitor=args.save_monitor+"/val_epoch",
        #     dirpath=dirpath,
        #     filename="topk/{epoch:d}-{step}-{accuracy/val_epoch:.4f}",
        #     save_top_k=args.save_top_k,
        #     mode='min' if args.save_monitor=='loss' else 'max',
        #     period=1,  # Check every validation epoch
        #     save_last=True,
        #     save_on_train_epoch_end=False,
        # )
    ]

    trainer = pl.Trainer(
        accumulate_grad_batches=args.accumulate,
        benchmark=args.benchmark,  # cudnn.benchmark
        callbacks=callbacks,
        deterministic=False,  # cudnn.deterministic
        gpus=args.gpus,
        gradient_clip_algorithm=args.grad_clip,
        gradient_clip_val=args.clip_value,
        logger=logger,
        precision=args.precision,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
        log_every_n_steps=1,
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_loader,
    )
