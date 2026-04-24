from src.trainer import BaseTrainer


class Trainer(BaseTrainer):
    def process_batch(self, batch: dict, metrics: list):
        batch = self._move_batch_to_device(batch)

        if self.is_train:
            self.optimizer.zero_grad()

        output = self.model(**batch)
        batch.update(output)
        loss = self.criterion(**batch)
        
        if self.is_train:
            loss.backward()
            self.optimizer.step()

        for metric in metrics:
            metric(**batch)
