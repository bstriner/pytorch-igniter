from torch.utils.data import IterableDataset, get_worker_info
from aws_sagemaker_remote.util.pipes import chunk_iterable, PipeIterator
from sagemaker.amazon.common import read_recordio
import os
from torch.multiprocessing import Value, Lock


class MultiprocessingCounter(object):
    def __init__(self):
        self.value = Value('i', 0)
        self.lock = Lock()
        self.local = 0


class PipeDataset(IterableDataset):
    # def init_fn(self, worker):
    #    print("Running init fn")
    #    self.count = count
    #    self.lock = lock

    def transform(self, rec, worker):
        if self.transform_fn:
            rec = self.transform_fn(rec, worker)
        return rec

    def __init__(self, paths, size, transform_fn=None, epochs=1):
        if isinstance(paths, dict):
            paths = list(paths.values())
        elif isinstance(paths, str):
            paths = [paths]
        self.paths = paths
        self.pipes = [
            PipeIterator(path, size=size, count=0, epochs=epochs)
            for path in paths
        ]
        self.size = size
        self.transform_fn = transform_fn
        super(PipeDataset, self).__init__()

    def __iter__(self):
        info = get_worker_info()
        if not info:
            worker = None
        else:
            worker = info.id
        if worker is None:
            assert len(self.paths) == 1
            worker = 0
        else:
            assert worker >= 0
            assert worker < len(self.paths)

        pipe = self.pipes[worker]
        for rec in pipe:
            rec = self.transform(rec, worker)
            yield rec
