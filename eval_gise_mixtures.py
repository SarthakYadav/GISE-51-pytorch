import os
import json
import glob
import numpy as np
import torch
import tqdm
from src.data.inmemory_lmdb_dataset import InMemorySpectrogramDataset
from src.data.utils import _collate_fn, _collate_fn_multiclass
from torch.utils.data import DataLoader
from src.data.transforms import get_transforms_v2
from src.models.gise_mixtures_lightning import GISEMixtures_Lightning
from src.utilities.metrics_helper import calculate_stats, d_prime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt_path", type=str,
                    help="path to model .ckpt")
parser.add_argument("--lbl_map", type=str,
                    help="path to label map .json file")
parser.add_argument("--lmdb_path")
parser.add_argument("--exp_dir", type=str, default=None)
parser.add_argument("--results_csv", type=str, default='results.csv')
parser.add_argument("--overlap_analysis", action="store_true")
parser.add_argument("--export_weights", action="store_true")
parser.add_argument("--export_dir", type=str)
parser.add_argument("--num_timesteps", type=int, default=501)


def eval_model(ckpt_path, val_tfs, overlap_analysis=False, export_weights=False, export_dir=None):
    # print(ckpt_path.split("/"))
    model_spec = ckpt_path.split("/")[-3]
    ckpt_ext = "/".join(ckpt_path.split("/")[-5:])
    # print(ckpt_ext)
    # print("Model: {}".format(model_spec))
    model = GISEMixtures_Lightning.load_from_checkpoint(ckpt_path)
    model = model.cuda().eval()

    test_set = InMemorySpectrogramDataset(args.lmdb_path, args.lbl_map,
                                          model.hparams.cfg['audio_config'],
                                          transform=val_tfs, is_val=True)
    data_loader = DataLoader(test_set, num_workers=8, batch_size=128,
                             shuffle=False, collate_fn=_collate_fn)
    test_predictions = []
    test_gts = []
    for batch in tqdm.tqdm(data_loader):
        x, _, y = batch
        x = x.cuda()
        with torch.no_grad():
            y_pred = model(x)
            sigmoid_preds = torch.sigmoid(y_pred)
        test_predictions.append(sigmoid_preds.detach().cpu())
        test_gts.append(y.detach().cpu())
    test_predictions = torch.cat(test_predictions, 0).numpy()
    test_gts = torch.cat(test_gts, 0).numpy()
    if overlap_analysis:
        with open("./metadata/overlapping_indices.json", 'r') as fd:
            indices = json.load(fd)
    else:
        indices = None
    stats = calculate_stats(test_predictions, test_gts, class_indices=indices)
    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    if export_weights:
        num_classes = model.hparams.cfg['model']['num_classes']
        splitted = ckpt_path.split("/")
        sub_dir = os.path.join(export_dir, splitted[-4], splitted[-3])
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        pth_path = os.path.join(sub_dir,
                        "mAP={:.4f}_dprime={:.4f}_num_classes={}.pth".format(mAP, d_prime(mAUC), num_classes)
                        )
        model.export_state_dict(pth_path)
    # print("mAP: {:.6f}".format(mAP))
    # print("mAUC: {:.6f}".format(mAUC))
    # print("dprime: {:.6f}".format(d_prime(mAUC)))
    return {
        "name": model_spec,
        "ckpt_ext": ckpt_ext,
        "mAP": mAP, "mAUC": mAUC, "dprime": d_prime(mAUC)
    }


if __name__ == '__main__':
    args = parser.parse_args()
    # model = FSD50kMixtures_Lightning.load_from_checkpoint(args.ckpt_path)
    # model = model.cuda().eval()
    val_tfs = get_transforms_v2(False, args.num_timesteps)
    if args.exp_dir is None:
        print(eval_model(args.ckpt_path, val_tfs))
    else:
        ckpts = glob.glob(os.path.join(args.exp_dir, "*", "*", "ckpts", "*.ckpt"))
        # print("num ckpts:", len(ckpts))
        results = []
        fd = open(args.results_csv, "w")
        fd.writelines("Model,mAP,mAUC,dprime,ckpt_path\n")
        for f in ckpts:
            res = eval_model(f, val_tfs, args.overlap_analysis, args.export_weights, args.export_dir)
            line = "{},{},{},{},{}\n".format(res['name'], res['mAP'],
                                                 res['mAUC'], res['dprime'],
                                                 res['ckpt_ext'])
            results.append(line)
            fd.writelines(line)
        fd.close()
        print("Results written to: ", args.results_csv)
