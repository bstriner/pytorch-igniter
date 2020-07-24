
from ignite.engine import Engine, Events
import numpy as np
from .util import tensors_to_numpy


class ConcatenateOutputsHook(object):
    def __init__(self, tracked_outputs, axis=-1):
        self.tracked_outputs = tracked_outputs
        self.axis = axis
        self.saved_outputs = {}

    def attach(self, engine: Engine,
               start_event=Events.ITERATION_COMPLETED,
               iteration_event=Events.ITERATION_COMPLETED
               ):
        if start_event is not None:
            engine.add_event_handler(
                event_name=start_event, handler=self.clear)
        if iteration_event is not None:
            engine.add_event_handler(
                event_name=iteration_event, handler=self.collect)

    def clear(self, engine):
        self.saved_outputs = {}

    def collect(self, engine):
        output = engine.state.output
        for k in self.tracked_outputs:
            if k not in self.saved_outputs:
                self.saved_outputs[k] = []
            self.saved_outputs[k].append(tensors_to_numpy(output[k]))

    def concatenate(self):
        return {
            k: np.concatenate(v, axis=0) for k, v in self.saved_outputs.items()
        }
