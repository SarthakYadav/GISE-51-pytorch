import os
import torch
import numpy as np
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.utils import _collate_fn, _collate_fn_multiclass
from src.models.model_helper import model_helper
from src.data.mixup import do_mixup, mixup_criterion


class ClassifierDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super(ClassifierDataModule, self).__init__()
        self.hparams = hparams
        self.mode = "multiclass"
        self.collate_fn = _collate_fn_multiclass

    def prepare_data(self):
        pass

    def setup(self, stage):
        is_lmdb = self.hparams.cfg['data'].get('is_lmdb', False)
        in_memory = self.hparams.cfg['data'].get('in_memory', True)
        if not is_lmdb:
            from src.data.dataset import SpectrogramDataset
        else:
            if in_memory:
                print("Using InMemorySpectrogramDataset (fetched from lmdb)")
                from src.data.inmemory_lmdb_dataset import InMemorySpectrogramDataset as SpectrogramDataset
            else:
                print("Using LMDBSpectrogramDataset")
                from src.data.lmdb_dataset import LMDBSpectrogramDataset as SpectrogramDataset

        self.train_set = SpectrogramDataset(self.hparams.cfg['data']['train'],
                                            self.hparams.cfg['data']['labels'],
                                            self.hparams.cfg['audio_config'],
                                            mode=self.mode, augment=True,
                                            mixer=self.hparams.tr_mixer,
                                            transform=self.hparams.tr_tfs, is_val=False)
        self.val_set = SpectrogramDataset(self.hparams.cfg['data']['val'],
                                          self.hparams.cfg['data']['labels'],
                                          self.hparams.cfg['audio_config'],
                                          mode=self.mode, augment=False,
                                          mixer=None,
                                          transform=self.hparams.val_tfs, is_val=True)

    def train_dataloader(self):
        return DataLoader(self.train_set, num_workers=self.hparams.num_workers, shuffle=True,
                          sampler=None, collate_fn=self.collate_fn,
                          batch_size=self.hparams.cfg['opt']['batch_size'],
                          pin_memory=False, drop_last=True, prefetch_factor=self.hparams.prefetch_factor)

    def val_dataloader(self):
        return DataLoader(self.val_set, sampler=None, num_workers=self.hparams.num_workers,
                          collate_fn=self.collate_fn,
                          shuffle=False, batch_size=self.hparams.cfg['opt']['batch_size'],
                          pin_memory=False)


class ClassifierPL(pl.LightningModule):
    def __init__(self, hparams):
        super(ClassifierPL, self).__init__()
        self.hparams = hparams
        self.net = model_helper(self.hparams.cfg['model'])
        assert self.hparams.cfg['model']['type'] == "multiclass"
        if self.hparams.cw is not None:
            print("Class weights found. Training weighted cross-entropy model")
            cw = torch.load(self.hparams.cw)
        else:
            print("Training weighted cross-entropy model")
            cw = None
        self.mode = "multiclass"
        self.collate_fn = _collate_fn_multiclass
        self.mixup_enabled = self.hparams.cfg['audio_config'].get("mixup", False)
        if self.mixup_enabled:
            print("Attention! Training with mixup")
            self.criterion = nn.CrossEntropyLoss(weight=cw, reduction="none")
        else:
            self.criterion = nn.CrossEntropyLoss(weight=cw)
        self.train_set = None
        self.val_set = None

        self.val_predictions = []
        self.val_gts = []

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_step):
        self.net.zero_grad()
        x, _, y = batch
        if self.mixup_enabled:
            mixed_x, y_a, y_b, lam = do_mixup(x, y, mode=self.mode)
            y_pred = self(x)
            loss = mixup_criterion(self.criterion, y_pred, y_a, y_b, lam)
            # print(loss)
        else:
            y_pred = self(x)
            loss = self.criterion(y_pred, y)
            # print(loss)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_step):
        x, _, y = batch
        y_pred = self(x)
        loss = self.criterion(y_pred, y)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        self.val_predictions.append(y_pred.detach().cpu().float())
        self.val_gts.append(y.detach().cpu())
        return loss

    def validation_epoch_end(self, outputs) -> None:
        val_preds = torch.cat(self.val_predictions, 0)
        val_gts = torch.cat(self.val_gts, 0)
        _, predicted = torch.max(val_preds.detach(), 1)
        correct = (predicted == val_gts).sum().item()
        acc = torch.tensor(correct / val_gts.size(0))
        self.log("val_acc", acc, prog_bar=True, on_epoch=True)
        self.val_predictions = []
        self.val_gts = []

    def configure_optimizers(self):
        wd = float(self.hparams.cfg['opt'].get("weight_decay", 0))
        lr = float(self.hparams.cfg['opt'].get("lr", 1e-3))
        optimizer_name = self.hparams.cfg['opt'].get("optimizer", "Adam")
        scheduler_name = self.hparams.cfg['opt'].get("scheduler", "reduce")
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=wd)
        elif optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), lr=lr, nesterov=True,
                                        momentum=0.9, weight_decay=wd)
        if scheduler_name == 'reduce':
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "max", factor=0.1,
                                                                      patience=5, verbose=True)
        elif scheduler_name == 'step':
            lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 10, gamma=0.1)
        to_monitor = "val_mAP" if self.mode == "multilabel" else "val_acc"
        return {
            "optimizer": optimizer,
            "lr_scheduler": lr_scheduler,
            "monitor": to_monitor
        }

    def export_state_dict(self, path):
        torch.save(self.net.eval().cpu().state_dict(), path)
