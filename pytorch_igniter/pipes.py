from torch.util.data import IterableDataset, get_worker_info
from aws_sagemaker_remote.util.pipes import chunk_iterable
from sagemaker.amazon.common import read_recordio


class PipeLoader(IterableDataset):
    def __init__(self, paths, size):
        self.paths = paths
        self.counts = {}
        self.size = size
        super(PipeLoader, self).__init__()

    def __iter__(self):
        info = get_worker_info()
        if not info:
            worker = None
        else:
            worker = info.id
        if worker is None:
            assert len(self.paths) == 1
            path = self.paths[0]
        else:
            assert worker >= 0
            assert worker < len(self.paths)
            path = self.paths[worker]
        if path not in self.counts:
            self.counts[path] = 0
        with open("{}_{}".format(path, self.counts[path]), 'rb') as f:
            self.counts[path] += 1
            for rec in chunk_iterable(read_recordio(f), self.size, last='error'):
                yield rec
