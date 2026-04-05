# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Callable, List, Optional, Tuple, Union

import numpy as np

from .extended import ExtendedVisionDataset
from glob import glob 

logger = logging.getLogger("dinov2")
_Target = int



class PD(ExtendedVisionDataset):
    Target = Union[_Target]


    def __init__(
        self,
        *,
        root: str,
        transforms: Optional[Callable] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)

        self._entries = glob('/home/work/dataset/test-cxr/train/*/*')#[(0,0,0),(1,1,1),(2,2,2),(3,3,3)]#None
        self._class_ids = None
        self._class_names = None


    def _get_entries(self) -> np.ndarray:
        if self._entries is None:
            self._entries = self._load_extra(self._entries_path)
        assert self._entries is not None
        return self._entries

    def _get_class_ids(self) -> np.ndarray:
        if self._class_ids is None:
            self._class_ids = self._load_extra(self._class_ids_path)
        assert self._class_ids is not None
        return self._class_ids

    def _get_class_names(self) -> np.ndarray:
        if self._class_names is None:
            self._class_names = self._load_extra(self._class_names_path)
        assert self._class_names is not None
        return self._class_names

    def find_class_id(self, class_index: int) -> str:
        class_ids = self._get_class_ids()
        return str(class_ids[class_index])

    def find_class_name(self, class_index: int) -> str:
        class_names = self._get_class_names()
        return str(class_names[class_index])

    def get_image_data(self, index: int) -> bytes:
        entries = self._get_entries()
        # actual_index = entries[index]["actual_index"]
        actual_index = index 
        
        class_id = self.get_class_id(index)

        image_relpath = self.split.get_image_relpath(actual_index, class_id)
        image_full_path = os.path.join(self.root, image_relpath)
        with open(image_full_path, mode="rb") as f:
            image_data = f.read()
        return image_data

    def get_target(self, index: int) -> Optional[Target]:
        entries = self._get_entries()
        class_index = entries[index]["class_index"]
        return int(class_index)

    def get_targets(self) -> Optional[np.ndarray]:
        entries = self._get_entries()
        return entries["class_index"]

    def get_class_id(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_id = entries[index]["class_id"]
        return str(class_id)

    def get_class_name(self, index: int) -> Optional[str]:
        entries = self._get_entries()
        class_name = entries[index]["class_name"]
        return str(class_name)

    def __len__(self) -> int:
        entries = self._get_entries()
        # assert len(entries) == self.split.length
        return len(entries)



    def _dump_entries(self) -> None:

        # NOTE: Using torchvision ImageFolder for consistency
        from torchvision.datasets import ImageFolder
        
        dataset_root = glob(self.root + '/') # /workspace/test-cxr/
        dataset = ImageFolder(dataset_root)
        sample_count = len(dataset)
        max_class_id_length, max_class_name_length = -1, -1
        for sample in dataset.samples:
            _, class_index = sample
            class_id, class_name = class_index, class_index
            max_class_id_length = max(len(class_id), max_class_id_length)
            max_class_name_length = max(len(class_name), max_class_name_length)

        dtype = np.dtype(
            [
                ("actual_index", "<u4"),
                ("class_index", "<u4"),
                ("class_id", f"U{max_class_id_length}"),
                ("class_name", f"U{max_class_name_length}"),
            ]
        )
        entries_array = np.empty(sample_count, dtype=dtype)

        labels = [(0,0), (1,1), (2,2), (3,3)]
        class_names = {class_id: class_name for class_id, class_name in labels}

        assert dataset
        old_percent = -1
        for index in range(sample_count):
            percent = 100 * (index + 1) // sample_count
            if percent > old_percent:
                logger.info(f"creating entries: {percent}%")
                old_percent = percent

            image_full_path, class_index = dataset.samples[index]
            image_relpath = os.path.relpath(image_full_path, self.root)
            class_id, actual_index = class_index, class_index
            class_name = class_names[class_id]
            entries_array[index] = (actual_index, class_index, class_id, class_name)

        logger.info(f'saving entries to "{self._entries_path}"')
        self._save_extra(entries_array, self._entries_path)

    def _dump_class_ids_and_names(self) -> None:
      
        entries_array = [(0,0,0),(1,1,1),(2,2,2),(3,3,3)]

        max_class_id_length, max_class_name_length, max_class_index = -1, -1, -1
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            max_class_index = max(int(class_index), max_class_index)
            max_class_id_length = max(len(str(class_id)), max_class_id_length)
            max_class_name_length = max(len(str(class_name)), max_class_name_length)

        class_count = max_class_index + 1
        class_ids_array = np.empty(class_count, dtype=f"U{max_class_id_length}")
        class_names_array = np.empty(class_count, dtype=f"U{max_class_name_length}")
        for entry in entries_array:
            class_index, class_id, class_name = (
                entry["class_index"],
                entry["class_id"],
                entry["class_name"],
            )
            class_ids_array[class_index] = class_id
            class_names_array[class_index] = class_name

        logger.info(f'saving class IDs to "{self._class_ids_path}"')
        self._save_extra(class_ids_array, self._class_ids_path)

        logger.info(f'saving class names to "{self._class_names_path}"')
        self._save_extra(class_names_array, self._class_names_path)

    def dump_extra(self) -> None:
        self._dump_entries()
        # self._dump_class_ids_and_names()
