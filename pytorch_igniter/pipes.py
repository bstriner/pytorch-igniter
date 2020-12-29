from torch.utils.data import IterableDataset, get_worker_info, DataLoader
from aws_sagemaker_remote.util.pipes import chunk_iterable, ProtobufPipeIterator, epoch_iterable
from sagemaker.amazon.common import read_recordio
import os
from torch.multiprocessing import Value, Lock, Queue, spawn, Process, get_context, Event, JoinableQueue
from queue import Empty, Full

MAX_FAIL = 30


def get_worker():
    info = get_worker_info()
    if not info:
        worker = 0
    else:
        worker = info.id
    assert worker >= 0
    return worker


class MultiprocessingSource(object):
    def __init__(self, iterable, maxsize, epochs=0, multiprocessing_context=None):
        self.iterable = iterable
        self.epochs = epochs
        self.maxsize = maxsize
        self.multiprocessing_context = multiprocessing_context or get_context()

    def run(self):
        #print("Running process")
        epochs = epoch_iterable(self.epochs)
        done = False
        sent = 0
        for _ in epochs:
            if done:
                break
            for example in self.iterable:
                if done:
                    break
                # print("Queueing item: {}".format({k: v.size()
                #                                  for k, v in example.items()}))
                failures = 0
                for v in example.values():
                    assert not v.requires_grad
                # if sent > self.maxsize:

                # todo: check performance gains
                """
                try:
                    buf = self.return_queue.get(block=False)
                    for k, v in example.items():
                        buf[k].copy_(v)
                    example = buf
                except Empty as e:
                    pass
                    #print("Return queue is empty")
                    #raise Exception("Return queue is empty") from e
                """
                while True:
                    if self.request_close.value:
                        #print("Got flag")
                        done = True
                        break
                    try:
                        self.queue.put(example, timeout=1)
                        sent += 1
                        break
                    except Full as e:
                        failures += 1
                        if failures >= MAX_FAIL:
                            raise Exception("Putting on queue failed") from e
                        else:
                            continue
        self.is_done.value = True
        self.done_event.wait(timeout=120)
        #print("Exiting process")

    def open(self):
        #print("Spawning process")
        self.queue = self.multiprocessing_context.Queue(
            maxsize=self.maxsize)
        # self.return_queue = self.multiprocessing_context.Queue(
        #    maxsize=self.maxsize)
        self.is_done = self.multiprocessing_context.Value('b')
        self.is_done.value = False
        self.request_close = self.multiprocessing_context.Value('b')
        self.request_close.value = False
        self.done_event = self.multiprocessing_context.Event()
        ctx = self.multiprocessing_context.Process(
            target=MultiprocessingSource.run,
            args=(self,),
            # nprocs=1,
            # join=False,  # True,
            # daemon=True,
            # start_method='spawn'
        )
        # print(ctx)
        self.ctx = ctx
        self.ctx.start()
        return ctx

    def close(self):
        self.request_close.value = True
        #self.is_done.value = True
        self.done_event.set()
        # self.queue.join()
        self.ctx.join()

    def __iter__(self):
        failures = 0
        while True:
            rec = None
            # if self.request_close.value:
            #    break
            try:
                rec = self.queue.get(timeout=1)
            except Empty:
                rec = None
            if rec is None:
                if self.is_done.value or failures >= MAX_FAIL:
                    break
                else:
                    failures += 1
                    continue
            #rec = self.transform(rec, worker)
            for v in rec.values():
                assert not v.requires_grad
            yield rec
            # self.queue.task_done()
            #self.return_queue.put(rec, block=False)
        if failures >= MAX_FAIL:
            raise Exception(
                f"Ran {MAX_FAIL} timeout failures without seeing is_done flag")


class PipeDataset(IterableDataset):
    def get_worker(self):
        return get_worker()

    # def transform(self, rec):
    #    return rec

    def __init__(
        self,
        path,
        num_workers,
        pipe_cls=ProtobufPipeIterator,
        multiprocessing_context=None,
        **pipe_args
    ):
        if isinstance(path, str):
            path = [path]
        self.path = path
        self.pipes = [
            pipe_cls(
                path=p,
                **pipe_args
            )
            for p in self.path
        ]
        self.mw = None
        if num_workers > 1:
            if len(path) == 1:
                self.mw = MultiprocessingSource(
                    self.pipes[0],
                    maxsize=20,  # todo,
                    epochs=1,
                    multiprocessing_context=multiprocessing_context
                )
            else:
                assert len(path) == num_workers
        else:
            assert len(path) == 1

    def __iter__(self):
        worker = self.get_worker()
        if self.mw:
            it = self.mw
        else:
            it = self.pipes[worker]
        return iter(it)
        # for i in it:
        #    yield i #self.transform(i)

    def open(self):
        if self.mw:
            self.mw.open()

    def close(self):
        if self.mw:
            self.mw.close()


class MultiprocessingCounter(object):
    def __init__(self):
        self.value = Value('i', 0)
        self.lock = Lock()
        self.local = 0


class ClosingDataLoader(DataLoader):
    def __iter__(self):
        self.dataset.open()
        for rec in super(ClosingDataLoader, self).__iter__():
            yield rec
        self.dataset.close()
