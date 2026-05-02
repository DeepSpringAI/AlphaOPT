import os

from omegaconf import OmegaConf


TRAIN_CONFIG_ENV = "ALPHAOPT_TRAIN_CONFIG"
DEFAULT_TRAIN_CONFIG = "train_config.yaml"


def get_train_config_path() -> str:
    return os.getenv(TRAIN_CONFIG_ENV, DEFAULT_TRAIN_CONFIG)


def load_train_config():
    return OmegaConf.load(get_train_config_path())
