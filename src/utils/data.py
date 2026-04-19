from torch.utils.data import DataLoader

from itertools import repeat
from math import ceil

from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.init import set_worker_seed
from src.dataset.collate_fn import collate_fn


def build_dataloaders(config: DictConfig):
    """
    Instantiates dataloaders
    """
    dataloaders = {}
    for part in config.dataset.keys():
        dataset = instantiate(config.dataset[part])
        dataloader = instantiate(
            config.dataloader,
            dataset=dataset,
            collate_fn=collate_fn,
            shuffle=(part == "train"),
            worker_init_fn=set_worker_seed
        )
        dataloaders[part] = dataloader
    
    return dataloaders

def dataloader_loop(dataloder: DataLoader, total_steps: int, epoch_len: int):
    n_epochs = ceil(total_steps / epoch_len)
    for epoch, dl in enumerate(repeat(dataloder, n_epochs)):
        for step, elem in enumerate(dl):
            yield elem
            if step + 1 >= epoch_len or epoch * epoch_len + step + 1 >= total_steps:
                break
