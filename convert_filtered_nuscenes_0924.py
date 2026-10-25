"""Run the MV2DFusion converter on one filtered nuScenes metadata chunk."""

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
