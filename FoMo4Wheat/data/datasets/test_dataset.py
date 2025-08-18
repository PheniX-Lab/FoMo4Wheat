# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

import csv
from enum import Enum
import logging
import os
from typing import Any, Callable, List, Optional, Tuple, Union

import numpy as np
from PIL import Image, UnidentifiedImageError

from .extended import ExtendedVisionDataset

logger = logging.getLogger("dinov2")
_Target = int


class TestDataset(ExtendedVisionDataset):

    def __init__(
            self,
            *,
            split: "ImageNet.Split",
            root: str,
            extra: str,
            transforms: Optional[Callable] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(root, transforms, transform, target_transform)
        self.root = root
        self.image_path_list = get_image_path_list(root)

    def get_image_data(self, index: int) -> Image:
        image_path = os.path.join(self.root, self.image_path_list[index])
        return image_path

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        try:
            image = Image.open(self.get_image_data(index)).convert("RGB")
        except (UnidentifiedImageError, OSError) as e:
            logger.warning(f"Skipping unreadable image: {self.get_image_data(index)}")
            return self.__getitem__((index + 1) % len(self))  # Skip and move to the next image

        if self.transforms is not None:
            image, _ = self.transforms(image, 1)
        return image, 1

    def __len__(self) -> int:
        return len(self.image_path_list)


def get_image_path_list(root):
    image_path_list = os.listdir(root)
    return image_path_list
