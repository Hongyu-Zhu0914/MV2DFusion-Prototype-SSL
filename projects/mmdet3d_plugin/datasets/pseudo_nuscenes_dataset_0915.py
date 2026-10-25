"""Opt-in nuScenes dataset for offline pseudo-label training.

This module is intentionally separate from ``nuscenes_dataset.py`` so that all
historical MV2DFusion configurations keep their original behaviour.  It loads
the supervised annotation file through the stock dataset, appends a second
pseudo-label PKL without globally re-sorting the two domains, and places an
``is_pseudo`` flag in every frame's ``img_metas``.
"""

import copy
import os.path as osp
import random

import mmcv
import numpy as np

from mmdet.datasets import DATASETS

from .nuscenes_dataset import CustomNuScenesDataset


@DATASETS.register_module()
class PseudoAwareNuScenesDataset0915(CustomNuScenesDataset):
    """A non-invasive supervised + offline-pseudo nuScenes dataset.

    Args:
        pseudo_ann_file (str, optional): Generated pseudo-label info PKL.  The
            regular ``ann_file`` remains the supervised source.
        require_pseudo_flag (bool): Require every appended pseudo record to
            explicitly contain ``is_pseudo=True``.

    The supervised and pseudo lists are sorted independently by timestamp and
    then concatenated.  Sorting the combined list globally is deliberately
    avoided because timestamps from different nuScenes releases may interleave
    unrelated scenes and break the temporal sampler's sequence boundaries.
    """

    def __init__(self, pseudo_ann_file=None, require_pseudo_flag=True, *args, **kwargs):
        self.pseudo_ann_file = pseudo_ann_file
        self.require_pseudo_flag = bool(require_pseudo_flag)
        super().__init__(*args, **kwargs)

        supervised_infos = []
        for index, raw_info in enumerate(self.data_infos):
            info = copy.copy(raw_info)
            if bool(info.get("is_pseudo", False)):
                raise ValueError(
                    f"supervised ann_file contains is_pseudo=True at index {index}"
                )
            info["is_pseudo"] = False
            supervised_infos.append(info)

        self.supervised_length = len(supervised_infos)
        self.pseudo_length = 0
        self.pseudo_metadata = None

        if pseudo_ann_file is None:
            self.data_infos = supervised_infos
            return

        pseudo_path = pseudo_ann_file
        if not osp.isabs(pseudo_path) and self.data_root is not None:
            pseudo_path = osp.join(self.data_root, pseudo_path)
        pseudo_data = mmcv.load(pseudo_path, file_format="pkl")
        if not isinstance(pseudo_data, dict) or not isinstance(pseudo_data.get("infos"), list):
            raise ValueError("pseudo_ann_file must contain {'infos': list, 'metadata': dict}")

        pseudo_infos = []
        for index, raw_info in enumerate(
                sorted(pseudo_data["infos"], key=lambda item: item["timestamp"])):
            if index % self.load_interval != 0:
                continue
            info = copy.copy(raw_info)
            if self.require_pseudo_flag and info.get("is_pseudo") is not True:
                raise ValueError(
                    f"pseudo_ann_file record {index} is missing explicit is_pseudo=True"
                )
            info["is_pseudo"] = True
            pseudo_infos.append(info)

        supervised_tokens = {str(info["token"]) for info in supervised_infos}
        pseudo_tokens = [str(info["token"]) for info in pseudo_infos]
        if len(set(pseudo_tokens)) != len(pseudo_tokens):
            raise ValueError("pseudo_ann_file contains duplicate sample tokens")
        overlap = supervised_tokens.intersection(pseudo_tokens)
        if overlap:
            raise ValueError(
                "supervised and pseudo annotation files overlap in sample tokens: "
                f"{sorted(overlap)[:3]}"
            )

        supervised_scenes = {str(info["scene_token"]) for info in supervised_infos}
        pseudo_scenes = {str(info["scene_token"]) for info in pseudo_infos}
        scene_overlap = supervised_scenes.intersection(pseudo_scenes)
        if scene_overlap:
            raise ValueError(
                "supervised and pseudo annotation files overlap in scenes: "
                f"{sorted(scene_overlap)[:3]}"
            )

        self.pseudo_metadata = copy.copy(pseudo_data.get("metadata", {}))
        self.pseudo_length = len(pseudo_infos)
        self.data_infos = supervised_infos + pseudo_infos

        # CustomNuScenesDataset created flags before the pseudo records were
        # appended.  Rebuild them over the final list for the temporal sampler.
        if self.seq_mode:
            self._set_sequence_group_flag()
        else:
            self._set_group_flag()

    @staticmethod
    def _attach_pseudo_flag(example, is_pseudo):
        if example is None:
            return None
        if "img_metas" not in example:
            raise KeyError("pipeline output does not contain img_metas")
        meta = example["img_metas"].data
        if not isinstance(meta, dict):
            raise TypeError(
                "expected per-frame img_metas.data to be a dict before union2one, "
                f"got {type(meta).__name__}"
            )
        meta["is_pseudo"] = bool(is_pseudo)
        return example

    def get_data_info(self, index):
        input_dict = super().get_data_info(index)
        input_dict["is_pseudo"] = bool(self.data_infos[index].get("is_pseudo", False))
        return input_dict

    def get_ann_info(self, index):
        ann_info = super().get_ann_info(index)
        # The upstream implementation creates np.array([]) as float64 for an
        # empty frame.  Classification losses expect integer class indices, so
        # normalize both empty and non-empty labels at the dataset boundary.
        ann_info["gt_labels_3d"] = np.asarray(
            ann_info["gt_labels_3d"], dtype=np.int64
        )
        return ann_info

    def prepare_train_data(self, index):
        """Run the stock pipeline and inject the flag before temporal union."""
        queue = []
        index_list = list(
            range(index - self.queue_length - self.random_length + 1, index)
        )
        random.shuffle(index_list)
        index_list = sorted(index_list[self.random_length:])
        index_list.append(index)
        prev_scene_token = None

        for frame_index in index_list:
            frame_index = max(0, frame_index)
            input_dict = self.get_data_info(frame_index)
            if not self.seq_mode:
                if input_dict["scene_token"] != prev_scene_token:
                    input_dict.update(dict(prev_exists=False))
                    prev_scene_token = input_dict["scene_token"]
                else:
                    input_dict.update(dict(prev_exists=True))

            self.pre_pipeline(input_dict)
            example = self.pipeline(input_dict)
            example = self._attach_pseudo_flag(example, input_dict["is_pseudo"])
            queue.append(example)

        for offset in range(self.num_frame_losses):
            if self.filter_empty_gt and (
                    queue[-offset - 1] is None
                    or ~(queue[-offset - 1]["gt_labels_3d"]._data != -1).any()):
                return None
        return self.union2one(queue)

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        self.pre_pipeline(input_dict)
        example = self.pipeline(input_dict)
        return self._attach_pseudo_flag(example, input_dict["is_pseudo"])
