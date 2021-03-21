import subprocess as sp
import os
import glob
from multiprocessing import Pool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--src_dir", type=str)
parser.add_argument("--tgt_dir", type=str)
parser.add_argument("--tgt_sample_rate", type=int, default=22050)
parser.add_argument("--move_txt", action="store_true")
parser.add_argument("--move_jams", action="store_true")
args = parser.parse_args()

files = glob.glob(os.path.join(args.src_dir, "*.wav"))
lf = len(files)
print(lf)
TGT_DIR = args.tgt_dir
SAMPLE_RATE = args.tgt_sample_rate
if not os.path.exists(TGT_DIR):
    os.makedirs(TGT_DIR)


def process_idx(idx):
    f = files[idx]
    src_txt = f.replace(".wav", ".txt")
    src_jams = f.replace(".wav", ".jams")
    fname = f.split("/")[-1].split(".")[0]
    tgt_f = os.path.join(TGT_DIR, "{}.flac".format(fname))
    tgt_txt = os.path.join(TGT_DIR, "{}.txt".format(fname))
    tgt_jams = os.path.join(TGT_DIR, "{}.jams".format(fname))
    command = "ffmpeg -nostats -loglevel 0 -i '{}' -ac 1 -af aformat=s16:{} '{}'".format(f, SAMPLE_RATE, tgt_f)
    # print(command)
    if not os.path.exists(tgt_f):
        sp.call(command, shell=True)

    if args.move_txt:
        txt_command = "cp '{}' '{}'".format(src_txt, tgt_txt)
        if not os.path.exists(tgt_txt):
            sp.call(txt_command, shell=True)
    if args.move_jams:
        jams_command = "cp '{}' '{}'".format(src_jams, tgt_jams)
        if not os.path.exists(tgt_jams):
            sp.call(jams_command, shell=True)
    if idx % 1000 == 0:
        print("Done {}/{}".format(idx, lf))


if __name__ == '__main__':
    pool = Pool(12)
    o = pool.map_async(process_idx, range(lf))
    res = o.get()
    pool.close()
    pool.join()
    # process_idx(0)
