import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    """
    Base class for loss
    """
    def __init__(self, name: str = None, *args, **kwargs):
        """
        Args:
            name (str): name of a loss
        """
        self.name = name if name else self.__str__()
        self._total = 0
        self._n = 0
        self._average = 0
        super().__init__()

    def update(self, *args, **kwargs):
        """
        Defines loss calculation.
        Must return `torch.Tensor` scalar
        """
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        value = self.update(*args, **kwargs)
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Loss returns not a Tensor, but {type(value)} type of value")

        self._total += value.item()
        self._n += 1
        self._average = self._total / self._n
        return value
        
    def result(self):
        """
        Returns loss values in `dict` format
        """
        return {
            "total": self._total,
            "n": self._n,
            "average": self._average
        }

    def avg(self):
        """
        Returns average loss value
        """
        return self._average

    def reset(self):
        """
        Sets all values to zero
        """
        self._total = 0
        self._n = 0
        self._average = 0
