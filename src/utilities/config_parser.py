import os
import yaml
from typing import Dict, Optional


def parse_config(config_file: str) -> Dict:
    with open(config_file, "r") as fd:
        cfg = yaml.load(fd, yaml.FullLoader)
    return cfg


def get_data_info(cfg: Dict, augment: Optional[bool] = True) -> Dict:
    try:
        print("[get_data_info]", cfg)
        meta_root = cfg['meta_root']
        train_manifest = cfg['train_manifest']
        val_manifest = cfg['val_manifest']
        label_map = cfg['label_map']
        is_lmdb = cfg.get("is_lmdb", False)
        if not is_lmdb:
            train_manifest = os.path.join(meta_root, train_manifest)
            val_manifest = os.path.join(meta_root, val_manifest)
            label_map = os.path.join(meta_root, label_map)
            in_memory = cfg.get("in_memory", False)

            results = {
                'train': train_manifest,
                "val": val_manifest,
                "labels": label_map,
                "in_memory": in_memory
            }

            test_manifest = cfg.get("test_manifest", None)
            if test_manifest and test_manifest != "None":
                test_manifest = os.path.join(meta_root, test_manifest)
                results["test"] = test_manifest
            results['bg_files'] = cfg.get("bg_files", None)
            print("[get_data_info]:", results)
        else:
            train_lmdb = cfg['train_lmdb']
            val_lmdb = cfg['val_lmdb']
            label_map = os.path.join(meta_root, label_map) if not os.path.exists(label_map) else label_map
            results = {
                'train': train_lmdb,
                "val": val_lmdb,
                "labels": label_map,
                "is_lmdb": True
            }
            test_lmdb = cfg.get("test_lmdb", None)
            if test_lmdb and test_lmdb != "None":
                results['test_lmdb'] = test_lmdb

        return results

    except KeyError as ex:
        print(ex)
        exit(-1)
