import os

import logging
import logging.config

import torch
import random
import numpy as np

from omegaconf import DictConfig
from pathlib import Path

from src.utils.io import ROOT_PATH, read_json


def set_random_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

def set_worker_seed(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def setup_logger_config(save_dir: Path, config_path: Path, resume: bool = False):
    if config_path.exists():
        logger_config = read_json(config_path)
        for _, handler in logger_config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])
        
        logging.config.dictConfig(logger_config)
    else:
        print(f"Warning: cannot find logger config in {str(config_path)}. Using basic config.")
        logging.basicConfig(level=logging.INFO, filemode="a" if resume else "w")

def setup_logger(config: DictConfig):
    save_dir = ROOT_PATH / config.trainer.save_dir / config.logging.run_name
    save_dir.mkdir(exist_ok=True, parents=True)
    logger_config_path = ROOT_PATH / "src" / "configs" / config.logging.config_path

    if config.trainer.resume_from:
        setup_logger_config(save_dir, logger_config_path, resume=True)
    else:
        setup_logger_config(save_dir, logger_config_path, resume=False)

    logger = logging.getLogger("train")
    logger.setLevel(logging.DEBUG)

    return logger
