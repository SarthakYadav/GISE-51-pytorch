# script used for original mixture generation
# can be used as a starting point for generating your own mixtures.

import scaper
from collections import Counter
import torch
import json
import glob
import tqdm
import numpy as np
import os
import argparse
import warnings
from collections import Counter

parser = argparse.ArgumentParser()
parser.add_argument("--fg_path", type=str)
parser.add_argument("--bg_path", type=str)
parser.add_argument("--out_folder", type=str)
parser.add_argument("--num_soundscapes", type=int, default=5000)
parser.add_argument("--max_events", type=int, default=3)
parser.add_argument("--duration", type=float, default=5.)
parser.add_argument("--lbl_map", type=str)
parser.add_argument("--cw", type=str)
parser.add_argument("--proc_idx", type=int, default=0)

args = parser.parse_args()

NUM_WORKERS = 6

# path_to_audio = os.path.expanduser('~/audio')
n_soundscapes = args.num_soundscapes
soundscape_duration = args.duration
min_events = 1
max_events = args.max_events

event_time_spec = ('uniform', 0.0, soundscape_duration)
event_duration_spec = ("uniform", 1.5, soundscape_duration)

pitch_spec = None

time_stretch_spec = ('uniform', 0.95, 1.05)
source_time_spec = ('uniform', 0.0, 140.0)       # max is larger than longest
snr_spec = ('uniform', 10, 60)

outfolder = args.out_folder

seed = 123

with open(args.lbl_map, "r") as fd:
    lbl_map = json.load(fd)

# cw = torch.load(args.cw)

files = np.asarray(glob.glob(os.path.join(args.fg_path, "*", "*.wav")))
files = files[np.random.permutation(len(files))]
labels = np.asarray([f.split("/")[-2] for f in files])

if args.cw is not None:
    cw = torch.load(args.cw)
else:
    mapped_labels = [lbl_map[y] for y in labels]
    cnt = Counter(mapped_labels)
    most_common = cnt.most_common()[0][1]
    ws = []
    for i in range(len(cnt)):
        ws.append(most_common / cnt[i])
    cw = torch.tensor(ws)
instance_weights = torch.tensor([cw[lbl_map[ix]] for ix in labels])

print(files[:5])
print(labels[:5])
print(instance_weights[:5])

sc = scaper.Scaper(soundscape_duration, args.fg_path, args.bg_path, random_state=seed)
sc.ref_db = -60
sc.sr = 22050
sc.protected_labels = []

if not os.path.exists(outfolder):
    os.makedirs(outfolder)

for n in tqdm.tqdm(range(n_soundscapes)):
    # print('Generating soundscape: {:d}/{:d}'.format(n+1, n_soundscapes))

    # reset the event specifications for foreground and background at the
    # beginning of each loop to clear all previously added events

    sc.reset_bg_event_spec()
    sc.reset_fg_event_spec()

    # add background
    sc.add_background(label=('choose', []),
                      source_file=('choose', []),
                      source_time=('uniform', 0, 10*60),)

    # add random number of foreground events
    n_events = np.random.randint(min_events, max_events+1)

    for ix in range(n_events):
        pick = torch.multinomial(instance_weights, 1, True)
        src_index = pick[0].item()
        sc.add_event(label=('const', labels[src_index]),
                     source_file=('const', files[src_index]),
                     source_time=source_time_spec,
                     event_time=event_time_spec,
                     event_duration=event_duration_spec,
                     snr=snr_spec,
                     pitch_shift=pitch_spec,
                     time_stretch=time_stretch_spec)

    # generate
    audiofile = os.path.join(outfolder, "soundscape_unimodal{:d}_proc_{:02d}.wav".format(n, args.proc_idx))
    jamsfile = os.path.join(outfolder, "soundscape_unimodal{:d}_proc_{:02d}.jams".format(n, args.proc_idx))
    txtfile = os.path.join(outfolder, "soundscape_unimodal{:d}_proc_{:02d}.txt".format(n, args.proc_idx))

    sc.generate(audiofile, jamsfile,
                allow_repeated_label=True,
                allow_repeated_source=True,
                reverb=None,
                fix_clipping=True,
                disable_sox_warnings=True,
                no_audio=False,
                txt_path=txtfile)
