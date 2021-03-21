import argparse
import numpy as np
import soundfile as sf
import tqdm
import glob
import os
import subprocess as sp

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str)
parser.add_argument("--tgt_dir", type=str)
parser.add_argument("--min_clip", type=float, default=1.25)
args = parser.parse_args()
files = glob.glob(os.path.join(args.src_dir, "*", "*.wav"))
print(files[:5])
lf = len(files)

TGT_DIR = args.tgt_dir
if not os.path.exists(TGT_DIR):
    os.makedirs(TGT_DIR)


def replicate_if_needed(x, min_clip_duration):
    if len(x) < min_clip_duration:
        tile_size = (min_clip_duration // x.shape[0]) + 1
        x = np.tile(x, tile_size)[:min_clip_duration]
    return x


def process_idx(idx):
    f = files[idx]
    # print(f)
    splitted = f.split("/")
    fname = splitted[-1].split(".")[0]
    lbl = splitted[-2]
    sub_fld = os.path.join(TGT_DIR, lbl)
    if not os.path.exists(sub_fld):
        os.makedirs(sub_fld)
    tgt_f = os.path.join(sub_fld, "{}.wav".format(fname))
    temp_f = os.path.join(sub_fld, "{}_temp.wav".format(fname))
    volume = 0.1
    if lbl in ['Whispering', "Wind", "Crumpling_and_crinkling_and_crushing", "Writing", "Cricket"]:
        volume = 0.025
    elif lbl in ['Glass', "Tap", "Zipper", "Rattle", "Chirp_and_tweet", "Tearing"]:
        volume = 0.05
    else:
        volume = 0.1
    # volume = 0.05 if lbl in ["Whispering", "Glass", "Tap", "Crumpling_and_crinkling_and_crushing", "Wind"] else 0.1
    command = "sox '{}' '{}' silence 1 {} 1% -1 {} 1%".format(f, temp_f, volume, volume)
    # print(command)
    sp.call(command, shell=True)
    x, sr = sf.read(temp_f)
    # dur = x.shape[0] / sr
    if x.shape[0] < 220:
        print("{} too short. deleting..".format(temp_f))
        os.remove(temp_f)
        return
    else:
        x = replicate_if_needed(x, int(sr*args.min_clip))
        sf.write(tgt_f, x, sr, "PCM_16")
        os.remove(temp_f)
    # if idx % 200 == 0:
    #   print("Done: {}/{}".format(idx, lf))


if __name__ == '__main__':
    # process_idx(0)
    for ix in tqdm.tqdm(range(lf)):
        process_idx(ix)
