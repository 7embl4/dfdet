from src.trainer import BaseTrainer
import time

class Trainer(BaseTrainer):
    def process_batch(self, batch: dict, metrics: list):
        batch = self._move_batch_to_device(batch)

        if self.is_train:
            self.optimizer.zero_grad()

        t1 = time.time()
        output = self.model(**batch)
        t2 = time.time()
        batch.update(output)
        loss = self.criterion(**batch)
        t3 = time.time()
        
        if self.is_train:
            loss.backward()
            self.optimizer.step()
        t4 = time.time()

        for metric in metrics:
            metric(**batch)
        t5 = time.time()
