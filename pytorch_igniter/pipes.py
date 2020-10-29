from torch.utils.data import IterableDataset, get_worker_info
from aws_sagemaker_remote.util.pipes import chunk_iterable
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

    def transform_fn(self, rec, worker):
        return rec

    def __init__(self, paths, size, transform=None, buffering=0):
        if isinstance(paths, dict):
            paths = list(paths.values())
        elif isinstance(paths, str):
            paths = [paths]
        self.paths = paths
        self.counters = [
            MultiprocessingCounter() for _ in range(len(paths))
        ]
        self.size = size
        self.transform = transform
        self.buffering = buffering
        super(PipeDataset, self).__init__()

    def __iter__(self):
        info = get_worker_info()
        if not info:
            worker = None
        else:
            worker = info.id
        if worker is None:
            assert len(self.paths) == 1
            path = self.paths[0]
            counter = self.counters[0]
            count = counter.local
            counter.local = count+1
        else:
            assert worker >= 0
            assert worker < len(self.paths)
            path = self.paths[worker]
            counter = self.counters[worker]
            with counter.lock:
                count = counter.value.value
                counter.value.value = count+1
        # if path not in self.counts:
        #    self.counts[path] = 0
        #if worker:
        #else:

        pipe_path = "{}_{}".format(path, count)
        print("Opening pipe on worker [{}]: {}".format(worker, pipe_path))
        if os.path.isdir(pipe_path):
            print("Is dir: {}".format(pipe_path))
        if os.path.isfile(pipe_path):
            print("Is file: {}".format(pipe_path))
        with open(pipe_path, 'rb', buffering=self.buffering) as f:
            for rec in chunk_iterable(read_recordio(f), self.size, last='error'):
                rec = self.transform_fn(rec, worker)
                if self.transform:
                    rec = self.transform(rec, worker)
                yield rec
            print("Pipe {} empty".format(pipe_path))
