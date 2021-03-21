import os
import lmdb
import tqdm
import msgpack
import msgpack_numpy
from src.data.utils import load_audio
import glob
import argparse
import io


def readfile(f):
    with open(f, "rb") as stream:
        return stream.read()


def get_label(f):
    with open(f.replace(".flac", ".txt"), "r") as fd:
        lines = fd.readlines()
    lbls = []
    for l in lines:
        lbls.append(l.strip().split("\t")[-1])
    return ",".join(lbls)


parser = argparse.ArgumentParser()
parser.add_argument("--mixture_dir", type=str)
parser.add_argument("--lmdb_path", type=str)
parser.add_argument("--write_frequency", type=int, default=1000)
parser.add_argument("--map_size", type=float, default=5e9)
parser.add_argument("--audio_min_duration", type=float, default=5.)


if __name__ == '__main__':
    args = parser.parse_args()
    dirs = args.mixture_dir.split(";")
    files = []
    for mix_dir in dirs:
        print("Reading: {}".format(mix_dir))
        fls = glob.glob(os.path.join(mix_dir, "*.flac"))
        files.extend(fls)
    # files = glob.glob(os.path.join(args.mixture_dir, "*.flac"))
    db = lmdb.open(args.lmdb_path, subdir=os.path.isdir(args.lmdb_path),
                   map_size=args.map_size, readonly=False,
                   meminit=False, map_async=True)
    txn = db.begin(write=True)
    for idx in tqdm.tqdm(range(len(files))):
        audio = readfile(files[idx])
        label = get_label(files[idx])
        txn.put(u'{}'.format(idx).encode('ascii'), msgpack.packb((audio, label), default=msgpack_numpy.encode))
        if idx % args.write_frequency == 0:
            txn.commit()
            txn = db.begin(write=True)
    # finish iterating through dataset
    txn.commit()
    keys = [u'{}'.format(k).encode('ascii') for k in range(len(files))]
    with db.begin(write=True) as txn:
        txn.put(b'__keys__', msgpack.packb(keys))
        txn.put(b'__len__', msgpack.packb(len(keys)))
    db.sync()
    db.close()
