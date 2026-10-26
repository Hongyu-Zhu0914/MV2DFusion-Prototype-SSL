"""Run the MV2DFusion converter on one filtered nuScenes metadata chunk."""

# Resolve imports and relative CLI paths after the publication-layout migration.
import sys as _bootstrap_sys
import os as _bootstrap_os
from pathlib import Path as _BootstrapPath
REPO_ROOT = _BootstrapPath(__file__).resolve().parents[2]
if str(REPO_ROOT) not in _bootstrap_sys.path:
    _bootstrap_sys.path.insert(0, str(REPO_ROOT))
if __name__ == '__main__':
    _bootstrap_os.chdir(str(REPO_ROOT))


import argparse

from tools.data_converter.nuscenes_converter import create_nuscenes_infos


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), required=True)
    args = parser.parse_args()
    create_nuscenes_infos(
        args.root,
        "nuscenes2d",
        version="v1.0-test" if args.split == "test" else "v1.0-trainval",
        max_sweeps=10,
    )


if __name__ == "__main__":
    main()
