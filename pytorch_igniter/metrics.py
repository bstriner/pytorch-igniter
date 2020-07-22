
from ignite.metrics import Average
import numpy as np
import numbers
import torch
from typing import Any, Callable, Optional, Union
from ignite.metrics.metric import Metric, reinit__is_reduced, sync_all_reduce


class SafeAverage(Average):
    @sync_all_reduce("accumulator", "num_examples")
    def compute(self) -> Union[Any, torch.Tensor, numbers.Number]:
        if self.num_examples < 1:
            return torch.from_numpy(np.array([np.nan], dtype=np.float32))
            # raise NotComputableError(
            #    "{} must have at least one example before" " it can be computed.".format(self.__class__.__name__)
            # )

        return self.accumulator / self.num_examples
