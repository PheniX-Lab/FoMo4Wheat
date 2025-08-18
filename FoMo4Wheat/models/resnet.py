from functools import partial
import math
import logging
from typing import Sequence, Tuple, Union, Callable,Optional

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.nn.init import trunc_normal_

from dinov2.layers import Mlp, PatchEmbed, SwiGLUFFNFused, MemEffAttention, NestedTensorBlock as Block


class DinoResnet(nn.Module):
    def __init__(self,
                 img_size: int=224,
                 in_chans: int=3,
                 stride: int = 1,
                 drop_path: Optional[nn.Module] = None,
                 interpolate_offset=0.1,
                 act_layer=nn.GELU,
                 block_fn=Block,
                 ffn_layer="mlp",
                 block_chunks=1,
                    ):
        
