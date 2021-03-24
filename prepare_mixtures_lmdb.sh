#!/bin/bash

MIXTURES_DIR=$1
LMDB_DIR=$2

echo "MIXTURES_DIR: ${MIXTURES_DIR}"
echo "LMDB_DIR: ${LMDB_DIR}"

echo "preparing train set with 5k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train_5k" --lmdb_path "${LMDB_DIR}/train_5k.lmdb" --map_size 2e10

echo "preparing train set with 10k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train_10k" --lmdb_path "${LMDB_DIR}/train_10k.lmdb" --map_size 2e10

echo "preparing train set with 15k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train_15k" --lmdb_path "${LMDB_DIR}/train_15k.lmdb" --map_size 2e10

echo "preparing train set with 20k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train_20k" --lmdb_path "${LMDB_DIR}/train_20k.lmdb" --map_size 2e10

echo "preparing train set with 30k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train_30k" --lmdb_path "${LMDB_DIR}/train_30k.lmdb" --map_size 2e10

echo "preparing train set with 40k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train_40k" --lmdb_path "${LMDB_DIR}/train_40k.lmdb" --map_size 2e10

echo "preparing train set with 50k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train_50k" --lmdb_path "${LMDB_DIR}/train_50k.lmdb" --map_size 2e10

echo "preparing MAIN training set with 60k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train" --lmdb_path "${LMDB_DIR}/train.lmdb" --map_size 2e10

echo "preparing train set with 70k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train;${MIXTURES_DIR}/train_p2" --lmdb_path ${LMDB_DIR}/train_70k.lmdb --map_size 2e10

echo "preparing train set with 80k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train;${MIXTURES_DIR}/train_p2;${MIXTURES_DIR}/train_p3" --lmdb_path ${LMDB_DIR}/train_80k.lmdb --map_size 2e10

echo "preparing train set with 90k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train;${MIXTURES_DIR}/train_p2;${MIXTURES_DIR}/train_p3;${MIXTURES_DIR}/train_p4" --lmdb_path ${LMDB_DIR}/train_90k.lmdb --map_size 2e10

echo "preparing train set with 100k soundscapes"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/train;${MIXTURES_DIR}/train_p2;${MIXTURES_DIR}/train_p3;${MIXTURES_DIR}/train_p4;${MIXTURES_DIR}/train_p5" --lmdb_path ${LMDB_DIR}/train_100k.lmdb --map_size 2e10

echo "preparing val set"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/val" --lmdb_path "${LMDB_DIR}/val.lmdb" --map_size 1e10

echo "preparing eval set"
python pack_mixtures_into_lmdb.py --mixture_dir "${MIXTURES_DIR}/eval" --lmdb_path "${LMDB_DIR}/eval.lmdb" --map_size 1e10

echo "Done"
