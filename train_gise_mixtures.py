import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from src.models.gise_mixtures_lightning import GISEMixtures_Lightning, GISEMixtureDataModule
from src.data.transforms import get_transforms_v2
from src.utilities.config_parser import parse_config, get_data_info


parser = argparse.ArgumentParser()
parser.description = "Training script for FSD50k baselines"
parser.add_argument("--cfg_file", type=str,
                    help='path to cfg file')
parser.add_argument("--expdir", "-e", type=str,
                    help="directory for logging and checkpointing")
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--cw", type=str, required=False,
                    help="path to serialized torch tensor containing class weights")
parser.add_argument("--resume_from", type=str,
                    help="checkpoint path to continue training from")
parser.add_argument('--mixer_prob', type=float, default=0.75,
                    help="background noise augmentation probability")
parser.add_argument("--fp16", action="store_true",
                    help='flag to train in FP16 mode')
parser.add_argument("--gpus", type=str, default="0",
                    help="Single or multiple gpus to train on. For multiple, use ', ' delimiter ")
parser.add_argument("--random_clip_size", type=int, default=500)
parser.add_argument("--val_clip_size", type=int, default=1000)
parser.add_argument("--use_mixers", action="store_true")
parser.add_argument("--use_mixup", action="store_true")
parser.add_argument("--prefetch_factor", type=int, default=4)


if __name__ == '__main__':
    args = parser.parse_args()
    print(args)

    args.output_directory = os.path.join(args.expdir, "ckpts")
    args.log_directory = os.path.join(args.expdir, "logs")

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)

    cfg = parse_config(args.cfg_file)
    data_cfg = get_data_info(cfg['data'])
    cfg['data'] = data_cfg
    args.cfg = cfg
    # ckpt_fd = "{}".format(args.output_directory) + "/{epoch:02d}_{train_mAP:.3f}_{val_mAP:.3f}"
    ckpt_fd = "{}".format(args.output_directory) + "/{epoch:02d}_{train_loss:.3f}_{val_mAP:.3f}"
    ckpt_callback = pl.callbacks.model_checkpoint.ModelCheckpoint(
        filepath=ckpt_fd,
        verbose=True, save_top_k=-1
    )
    es_cb = pl.callbacks.EarlyStopping("val_mAP", mode="max", verbose=True, patience=10)
    # can't get it to work with val_mAP for now, loss should do too
    prog = pl.loggers.CSVLogger(args.log_directory)
    tb_prog = TensorBoardLogger(args.log_directory)
    loggers = [prog, tb_prog]
    if args.use_mixers:
        raise DeprecationWarning("running with --use_mixers is deprecated. Not doing anything")
        exit(1)
        # mixer = mixers.RandomMixer([
        #         mixers.SigmoidConcatMixer(sigmoid_range=(3, 12)),
        #         mixers.AddMixer(alpha_dist="uniform")
        #     ], p=[0.6, 0.4])
        # # mixer = mixers.BackgroundAddMixer()
        # mixer = mixers.UseMixerWithProb(mixer, args.mixer_prob)
        # args.tr_mixer = mixer    # mixers.UseMixerWithProb(mixer, args.mixer_prob)
    else:
        args.tr_mixer = None
    tr_tfs = get_transforms_v2(True, args.random_clip_size)
    val_tfs = get_transforms_v2(False, args.val_clip_size)

    args.tr_tfs = tr_tfs
    args.val_tfs = val_tfs

    # net = FSD50k_Lightning(args)
    net = GISEMixtures_Lightning(args)
    data_module = GISEMixtureDataModule(args)
    precision = 16 if args.fp16 else 32
    trainer = pl.Trainer(gpus=args.gpus, max_epochs=args.epochs, progress_bar_refresh_rate=1,
                         precision=precision, accelerator="ddp", prepare_data_per_node=False,
                         # num_sanity_val_steps=4170,
                         callbacks=[ckpt_callback, es_cb], replace_sampler_ddp=False,
                         resume_from_checkpoint=args.resume_from,
                         logger=[tb_prog, prog])
    trainer.fit(net, datamodule=data_module)
