import torch
import numpy as np


class BaseMetric:
    """
    Base class for metrics. 
    Stores value of a metric and calculates it's average
    """
    def __init__(self, name: str = None, *args, **kwargs):
        """
        Args:
            name (str): name of a metric
        """
        self.name = name if name else self.__str__()
        self._total = 0
        self._n = 0
        self._average = 0

    def update(self, *args, **kwargs):
        """
        Defines metric calculation.
        Must return scalar value or `torch.Tensor`/`numpy.ndarray` scalar
        """
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        """
        Calculates metric value and stores it
        """
        value = self.update(*args, **kwargs)
        if isinstance(value, torch.Tensor) or isinstance(value, np.ndarray):
            value = value.item()

        if value is not None:
            self._total += value
            self._n += 1
            self._average = self._total / self._n

    def result(self):
        """
        Returns metric values in `dict` format
        """
        return {
            "total": self._total,
            "n": self._n,
            "average": self._average
        }

    def avg(self):
        """
        Returns average metric value
        """
        return self._average

    def reset(self):
        """
        Sets all values to zero
        """
        self._total = 0
        self._n = 0
        self._average = 0
