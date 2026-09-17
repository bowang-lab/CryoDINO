# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import argparse
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup_3d
import dinov2.utils.utils as dinov2_utils


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--allow-random-init",
        action="store_true",
        help="Proceed with a RANDOMLY initialized backbone if --pretrained-weights is missing. "
             "Off by default: a silently random frozen backbone looks exactly like a model that "
             "will not learn.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights, allow_random_init=False):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    try:
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    except FileNotFoundError:
        # AA: this used to print and carry on, which silently trains a frozen RANDOM backbone —
        # indistinguishable from "the model won't learn" in a long log. Fail loudly instead;
        # pass --allow-random-init to opt in deliberately (e.g. an ablation).
        if not allow_random_init:
            raise FileNotFoundError(
                f"Pretrained weights not found: {pretrained_weights}. The backbone would be "
                f"randomly initialized and the model could not learn. Pass --allow-random-init "
                f"if that is genuinely what you want."
            )
        print(f"No weights found at {pretrained_weights}; --allow-random-init given, "
              f"continuing with RANDOM initialization.", flush=True)
    model.eval()
    model.cuda()
    return model


def setup_and_build_model_3d(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup_3d(args)
    model = build_model_for_eval(config, args.pretrained_weights,
                                 allow_random_init=getattr(args, 'allow_random_init', False))
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype
