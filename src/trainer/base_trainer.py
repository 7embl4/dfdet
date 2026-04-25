import logging

import torch
import torch.nn as nn

import numpy as np
from sklearn.metrics import f1_score, roc_auc_score

from omegaconf import DictConfig
from tqdm import tqdm

from src.loss import BaseLoss
from src.metrics import BaseMetric
from src.utils.io import ROOT_PATH
from src.utils.data import dataloader_loop


class BaseTrainer:
    def __init__(
        self,
        config: DictConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: BaseLoss,
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
        dataloaders: dict,
        train_metrics: list,
        inference_metrics: list,
        logger: logging.Logger,
        device: str
    ):
        self.config: DictConfig = config
        self.trainer_config: DictConfig = config.trainer
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr_scheduler = lr_scheduler
        self.train_metrics = train_metrics
        self.inference_metrics = inference_metrics
        self.logger = logger
        self.device = device

        # setup dataloaders
        self.train_dataloader = dataloaders["train"]
        self.evaluation_dataloaders = {
            part: dl for part, dl in dataloaders.items() if part != "train" 
        }

        # training strategy
        self.strategy = self.trainer_config.strategy
        assert self.strategy in ["epochs", "steps"], ValueError(f"Unknown strategy {self.strategy}")
        

        # train loop
        self.epoch_len = len(self.train_dataloader)
        if self.strategy == "epochs":
            if self.trainer_config.get("epoch_len", None):
                self.epoch_len = self.trainer_config.epoch_len
            self.total_steps = self.trainer_config.n_epochs * self.epoch_len
        else:
            self.total_steps = self.trainer_config.n_steps

        # evaluation
        self.eval_steps = self.trainer_config.get("eval_steps", None)
        if not self.eval_steps:
            self.eval_steps = self.epoch_len
        
        self.train_loop = tqdm(
            dataloader_loop(self.train_dataloader, self.total_steps, self.epoch_len),
            desc="train",
            total=self.total_steps,
        )

        # early stopping
        self.early_stop = self.trainer_config.early_stopping
        if self.early_stop:
            self.mnt_mode = self.trainer_config.mode
            assert self.mnt_mode in ["min", "max"], ValueError(f"Unknown mode {self.mnt_mode}")
            
            self.mnt_best = float("inf") if self.mnt_mode == "min" else float("-inf")
            self.mnt_name = self.trainer_config.value_name
            if self.strategy == "epochs":
                self.mnt_patience = self.trainer_config.patience * self.epoch_len
            else:
                self.mnt_patience = self.trainer_config.patience

            self.not_improved_count = 0

        # saving and logging
        self._last_step = 0
        self.output_dir = ROOT_PATH / self.trainer_config.save_dir / config.logging.run_name
        if self.strategy == "epochs":
            self.save_period = self.trainer_config.save_period * self.epoch_len
        else:
            self.save_period = self.trainer_config.save_period

    def train(self):
        try:
            self._train_process()
        except KeyboardInterrupt as e:
            self.logger.info("Saving model on keyboard interruption")
            self._save_checkpoint(self._last_step, best=False)
            raise e

    def _train_process(self):        
        self.is_train = True
        self.model.train()
        self._reset_metrics(self.train_metrics)
        for step, batch in enumerate(self.train_loop):
            self._last_step = step + 1
            batch = self.process_batch(batch, metrics=self.train_metrics)

            if (step + 1) % self.eval_steps == 0:
                logs = {f"train_{metric.name}": metric.avg() for metric in self.train_metrics}
                logs.update({f"grad_norm": self._calc_grad_norm()})
                logs.update({f"train_{self.criterion.name}": self.criterion.avg()})
                self._reset_metrics(self.train_metrics)
                
                logs.update(self._evaluate(step, self.total_steps))

                # TODO: make better output
                for key, value in logs.items():
                    self.logger.info(f"    {key:20s}: {value}")

                # TODO: add writing to tb, cometml

                best, early_stop = self._monitor_performance(logs)
                
                if self.trainer_config.get("clear_cache", None):
                    torch.cuda.empty_cache()

                if (step + 1) % self.save_period == 0 or best:
                    self._save_checkpoint(step + 1, best)

                if early_stop:
                    break
    
    def _evaluate(self, step: int, total_steps: int):
        self.is_train = False
        self.model.eval()

        eval_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            self._evaluation_epoch(part, dataloader)
            eval_logs.update({f"{part}_{metric.name}": metric.avg() for metric in self.inference_metrics})
            eval_logs.update({f"{part}_{self.criterion.name}": self.criterion.avg()})

        self.model.train()
        self.is_train = True

        if self.strategy == "epoch":
            eval_logs.update({"epoch": total_steps / (step + 1)})
        else:
            eval_logs.update({"step": step + 1})

        return eval_logs

    @torch.no_grad()
    def _evaluation_epoch(self, part, dataloader):
        self._reset_metrics(self.inference_metrics)
        eval_loop = tqdm(
            enumerate(dataloader),
            desc=part,
            total=len(dataloader),
            leave=False
        )

        for _, batch in eval_loop:
            batch = self.process_batch(batch, metrics=self.inference_metrics)

    def _monitor_performance(self, logs: dict):
        best = False
        stop_process = False
        try:
            if self.mnt_mode == "min":
                improved = logs[self.mnt_name] <= self.mnt_best
            else:
                improved = logs[self.mnt_name] >= self.mnt_best
        except KeyError:
            self.logger.warning(
                f"""
                Warning: Metric or Loss {self.mnt_name} is not found.
                Early stop is disabled.
                """
            )
            self.early_stop = False
        
        if improved:
            self.mnt_best = logs[self.mnt_name]
            self.not_improved_count = 0
            best = True
        else:
            self.not_improved_count += 1
        
        if self.early_stop:
            if self.not_improved_count >= self.mnt_patience:
                self.logger.info("Stopping on early stop")
                stop_process = True
        
        return best, stop_process

    def process_batch(self, batch: dict, metrics: list):
        raise NotImplementedError()

    def _reset_metrics(self, metrics: list[BaseMetric]):
        for metric in metrics:
            metric.reset()

    def _move_batch_to_device(self, batch: dict):
        for t in self.trainer_config.device_tensors:
            batch[t] = batch[t].to(self.device)
        return batch

    def _clip_grad_norm(self):
        if self.trainer_config.max_grad_norm:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.trainer_config.max_grad_norm)

    def _calc_grad_norm(self, norm_type: int = 2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad.detach(), norm_type) for p in parameters]),
            norm_type,
        )
        return total_norm.item()

    def _save_checkpoint(self, step, best):
        state = {
            "arch": type(self.model).__name__,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": self.config,
        }

        if self.strategy == "epoch":
            epoch = step // self.epoch_len
            state["epoch"] = epoch
        else:
            state["step"] = step

        epoch_or_step = state.get("epoch", step)
        if best:
            filename = str(self.output_dir / "model_best.pth")
        else:
            filename = str(self.output_dir / f"checkpoint-{epoch_or_step}.pth")
        
        torch.save(state, filename)
