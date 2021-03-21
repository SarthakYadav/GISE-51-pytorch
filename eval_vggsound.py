import sys
from src.models.classifier_lightning import ClassifierPL, ClassifierDataModule
from src.data.transforms import get_transforms_classifier
from src.data.inmemory_lmdb_dataset import InMemorySpectrogramDataset
import torch
import numpy as np
import soundfile as sf
import json
from src.dataset import load_audio
from src.data.audio_parser import AudioParser
from torch.utils.data import DataLoader
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from src.data.utils import _collate_fn, _collate_fn_multiclass
from src.utilities.metrics_helper import calculate_stats, d_prime
from scipy import signal
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str,
                    help="path to model .ckpt")
parser.add_argument("--lbl_map", type=str,
                    help="path to label map .json file")
parser.add_argument("--lmdb_path")


def one_hot_encode(y, n_classes=309):
    out = torch.zeros(len(y), n_classes)
    for i in range(len(y)):
        out[i, y[i]] = 1
    return out


if __name__ == '__main__':
    args = parser.parse_args()
    model = ClassifierPL.load_from_checkpoint(args.ckpt_path)
    val_tfs = get_transforms_classifier(False, 1001, center_crop_val=True)
    dset = InMemorySpectrogramDataset(args.lmdb_path, args,lbl_map, model.hparams.cfg['audio_config'], 
                                  mode="multiclass", augment=False, mixer=None, transform=val_tfs, is_val=True)
    loader = DataLoader(dset, sampler=None, num_workers=4,
                    collate_fn=_collate_fn_multiclass,
                    shuffle=False, batch_size=128,
                    pin_memory=False)
    model = model.cuda().eval().float()
    preds = []
    gts = []
    ix = 0
    for batch in loader:
        x, _, y = batch
        with torch.no_grad():
            x = x.cuda()
            # y = y.cuda()
            o = model(x)
            # print(o.shape)
        preds.append(o)
        gts.append(y)
        ix += 1
        if ix % 5 == 0:
            print("done:", ix)
    preds = torch.cat(preds)
    gts = torch.cat(gts)

    _, predicted = torch.max(preds.detach().cpu(), 1)
    _, topk_predicted = torch.topk(preds.detach(), 3, 1)
    acc = (predicted == gts).sum().item() / gts.size(0)

    stats = calculate_stats(torch.softmax(preds, 1).detach().cpu().numpy(), one_hot_encode(gts, 309).detach().cpu().numpy())
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    print("mAP: {} | acc:{} | AUC:{} | dprime: {}".format(mAP, acc, mAUC, d_prime(mAUC)))
