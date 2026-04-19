import torch

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig

from src.utils.init import set_random_seed, setup_logger
from src.utils.data import build_dataloaders
from src.trainer import Trainer


@hydra.main(version_base=None, config_path="src/configs", config_name="train")
def main(config: DictConfig):
    # set seed
    set_random_seed(config.seed)

    # get device
    if config.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = config.device

    # setup logger
    logger = setup_logger(config)

    # get dataloaders
    dataloaders = build_dataloaders(config)

    # build metrics
    metrics = instantiate(config.metrics)
    train_metrics, inference_metrics = metrics["train"], metrics["inference"]

    # build model
    model = instantiate(config.model)
    model.to(device)

    # instanciate criterion, optim, and scheduler
    criterion = instantiate(config.loss)
    optimizer = instantiate(config.optimizer, params=model.parameters())
    lr_scheduler = instantiate(config.lr_scheduler, optimizer=optimizer)

    # trainer
    trainer = Trainer(
        config,
        model,
        optimizer,
        criterion,
        lr_scheduler,
        dataloaders,
        train_metrics,
        inference_metrics,
        logger,
        device
    )
    trainer.train()


if __name__ == "__main__":
    main()
