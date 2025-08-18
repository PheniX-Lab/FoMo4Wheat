# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

from .train import get_args_parser, main
from .ssl_meta_arch import SSLMetaArch
from .train_distill import main as main_distill
from .load_cp import main as main_load_cp
from .distill_meta_arch import DistillMetaArch
