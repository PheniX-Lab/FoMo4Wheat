import  dinov2.eval.setup
import argparse
import OmegaConf
import pathlib
import os

def load_config(config_name: str):
    config_filename = config_name + ".yaml"
    return OmegaConf.load(pathlib.Path(__file__).parent.resolve() / config_filename)

def get_cfg_from_args(args):
    dinov2_default_config = load_config("ssl_default_config")
    cfg = OmegaConf.load(args.config_file)
    default_cfg = OmegaConf.create(dinov2_default_config)
    cfg = OmegaConf.merge(default_cfg, cfg, OmegaConf.from_cli(args.opts))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg_from_args(args)

    return cfg


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained_weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        help="Config file path",
    )
    parser.add_argument(
        "--opt",

    )


    return parser



args = get_args_parser()
config = setup(args)
dinov2.eval.setup.build_model_for_eval(config, args.pretrained_weights)