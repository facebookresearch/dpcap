# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import os
import random
import sys
from dataclasses import dataclass
from multiprocessing import Value
import torch

import braceexpand
import webdataset as wds
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
from torch.utils.data.distributed import DistributedSampler
from webdataset.filters import _shuffle
from webdataset.tariterators import base_plus_ext, url_opener, valid_sample, group_by_keys, meta_prefix, meta_suffix
import webdataset.filters as filters
from webdataset.handlers import reraise_exception
import re, tarfile
# from .open_clip import tokenize


class SharedEpoch:
    def __init__(self, epoch: int = 0):
        self.shared_epoch = Value('i', epoch)

    def set_value(self, epoch):
        self.shared_epoch.value = epoch

    def get_value(self):
        return self.shared_epoch.value


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


def preprocess_txt(text):
    return [str(text)]
    # return tokenize([str(text)])[0]


def get_dataset_size(shards, nb_samples=None):
    assert nb_samples is not None
    shards_list = list(braceexpand.braceexpand(shards))
    total_size = nb_samples
    # some common dataset sizes
    # CC3M (train): 2905954
    # CC12M: 10968539
    # LAION-400M: 407332084
    # LAION-233M: 233804767
    # LAION-2B (english): 2170337258
    num_shards = len(shards_list)
    return total_size, num_shards


def count_samples(dataloader):
    os.environ["WDS_EPOCH"] = "0"
    n_elements, n_batches = 0, 0
    for images, texts in dataloader:
        n_batches += 1
        n_elements += len(images)
        assert len(images) == len(texts)
    return n_elements, n_batches

def filter_no_caption_or_no_image(sample):
    return ('txt' in sample) and ('png' in sample or 'jpg' in sample)


def log_and_continue(exn):
    """Call in an exception handler to ignore any exception, isssue a warning, and continue."""
    logging.warning(f'Handling webdataset error ({repr(exn)}). Ignoring.')
    return True

######################################
######################################
def group_by_keys_nothrow(data, keys=base_plus_ext, lcase=True, suffixes=None, handler=None):
    """Return function over iterator that groups key, value pairs into samples.
    :param keys: function that splits the key into key and extension (base_plus_ext)
    :param lcase: convert suffixes to lower case (Default value = True)
    """
    current_sample = None
    for filesample in data:
        assert isinstance(filesample, dict)
        fname, value = filesample["fname"], filesample["data"]
        prefix, suffix = keys(fname)
        if prefix is None:
            continue
        if lcase:
            suffix = suffix.lower()
        # FIXME webdataset version throws if suffix in current_sample, but we have a potential for
        #  this happening in the current LAION400m dataset if a tar ends with same prefix as the next
        #  begins, rare, but can happen since prefix aren't unique across tar files in that dataset
        if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
            if valid_sample(current_sample):
                yield current_sample
            current_sample = dict(__key__=prefix, __url__=filesample["__url__"], __path__=filesample["__url__"].split("/")[-1].split(".")[0]+"/"+fname)
        if suffixes is None or suffix in suffixes:
            current_sample[suffix] = value
    if valid_sample(current_sample):
        yield current_sample


# def tarfile_to_samples_nothrow(src, handler=log_and_continue):
#     # NOTE this is a re-impl of the webdataset impl with group_by_keys that doesn't throw
#     streams = url_opener(src, handler=handler)
#     files = tar_file_expander(streams, handler=handler)
#     samples = group_by_keys_nothrow(files, handler=handler)
#     return samples

######################################
######################################
# https://github.com/webdataset/webdataset/blob/main/webdataset/tariterators.py

def trie_search(trie, seq='00000/00000'):
  try:
    if trie[seq]:
      return True
  except KeyError:
      return False


def tar_file_iterator(fileobj, url, trie, skip_meta=r"__[^/]*__($|/)", handler=reraise_exception):
    """Iterate over tar file, yielding filename, content pairs for the given tar stream.
    :param fileobj: byte stream suitable for tarfile
    :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")
    """
    # open the tar file
    stream = tarfile.open(fileobj=fileobj, mode="r|*")

    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            if (
                "/" not in fname
                and fname.startswith(meta_prefix)
                and fname.endswith(meta_suffix)
            ):
                # skipping metadata for now
                continue
            if skip_meta is not None and re.match(skip_meta, fname):
                continue

            ###############
            ###############

            file_key = url.split('.')[0].split('/')[-1]+'/'+fname.split('.')[0]

            if trie is not None and not trie_search(trie, file_key): 
                print(f"trie id {id(trie)}", "❌❌ skip ", file_key, url, fname)
                continue
            # ✅✅ Read the file here (if you pass from the if statement)
            data = stream.extractfile(tarinfo).read()
            result = dict(fname=fname, data=data)
            yield result
            ###############
            ###############


            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (exn.args[0] + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream


def tar_file_expander(data, trie, handler=reraise_exception):
    """Expand a stream of open tar files into a stream of tar file contents.
    This returns an iterator over (filename, file_contents).
    """
    for source in data:
        # tar file source. Each source item has "url" key and "stream" key
        url = source["url"] # shard url
        
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator(source["stream"], url, trie=trie):
                # Sample from tar file. Each sample has "fname" key and "data" key
                assert (
                    isinstance(sample, dict) and "data" in sample and "fname" in sample
                )
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                break



def tarfile_to_samples_nothrow(src, trie, handler=log_and_continue):
    '''
    streams_of_tar_files = url_opener ----> tar_file_expander(streams) -----> tar_file_iterator() -------> group_by_keys()
    '''
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, trie=trie, handler=handler)
    samples = group_by_keys_nothrow(files, handler=handler)
    
    return samples

def tarfile_to_samples(src, trie, handler=log_and_continue):
    '''
    streams_of_tar_files = url_opener ----> tar_file_expander(streams) -----> tar_file_iterator() -------> group_by_keys()
    '''
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, trie=trie, handler=handler)
    samples = group_by_keys(files, handler=handler)
    
    return samples

# tarfile_to_samples_filter = filters.pipelinefilter(tarfile_to_samples)
# print(type(tarfile_to_samples_filter))
# print(type(tarfile_to_samples_filter(mytrie)))
######################################
######################################


def pytorch_worker_seed():
    """get dataloader worker seed from pytorch"""
    worker_info = get_worker_info()
    if worker_info is not None:
        # favour the seed already created for pytorch dataloader workers if it exists
        return worker_info.seed
    # fallback to wds rank based seed
    return wds.utils.pytorch_worker_seed()


_SHARD_SHUFFLE_SIZE = 2000
_SHARD_SHUFFLE_INITIAL = 500
_SAMPLE_SHUFFLE_SIZE = 5000
_SAMPLE_SHUFFLE_INITIAL = 1000


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch = epoch

    def run(self, src):
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        rng = random.Random()
        if self.seed < 0:
            seed = pytorch_worker_seed() + epoch
        else:
            seed = self.seed + epoch
        rng.seed(seed)
        return _shuffle(src, self.bufsize, self.initial, rng)


class ResampledShards2(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(
        self,
        urls,
        nshards=sys.maxsize,
        worker_seed=None,
        deterministic=False,
        epoch=-1,
    ):
        """Sample shards from the shard list with replacement.
        :param urls: a list of URLs as a Python list or brace notation string
        """
        super().__init__()
        urls = wds.shardlists.expand_urls(urls)
        self.urls = urls
        assert isinstance(self.urls[0], str)
        self.nshards = nshards
        self.rng = random.Random()
        self.worker_seed = pytorch_worker_seed if worker_seed is None else worker_seed
        self.deterministic = deterministic
        self.epoch = epoch

    def __iter__(self):
        """Return an iterator over the shards."""
        if isinstance(self.epoch, SharedEpoch):
            epoch = self.epoch.get_value()
        else:
            # NOTE: this is epoch tracking is problematic in a multiprocess (dataloader workers or train)
            # situation as different workers may wrap at different times (or not at all).
            self.epoch += 1
            epoch = self.epoch
        if self.deterministic:
            # reset seed w/ epoch if deterministic, worker seed should be deterministic due to arg.seed
            self.rng.seed(self.worker_seed() + epoch)
        for _ in range(self.nshards):
            yield dict(url=self.rng.choice(self.urls))


def get_wds_dataset(
    batch_size,
    input_shards=None,
    preprocess_img=None,
    epoch=0,
    floor=False,
    resampled=False,
    world_size=1,
    num_workers=1,
    return_txt=True,
    trie=None,
    nb_samples=None
):

    assert input_shards is not None
    # print('✅ get_wds_dataset')
    # import ctypes
    # trie = ctypes.cast(trie, ctypes.py_object).value
    # print('✅ trie copied')

    num_samples, num_shards = get_dataset_size(input_shards, nb_samples=nb_samples)

    shared_epoch = SharedEpoch(epoch=epoch)  # create a shared epoch store to sync epoch to dataloader worker proc
    if resampled:
        pipeline = [ResampledShards2(input_shards, deterministic=True, epoch=shared_epoch)]
    else:
        pipeline = [wds.SimpleShardList(input_shards)]

    # at this point we have an iterator over all the shards
    if not resampled:
        pipeline.extend([
            detshuffle2(
                bufsize=_SHARD_SHUFFLE_SIZE,
                initial=_SHARD_SHUFFLE_INITIAL,
                seed=0,
                epoch=shared_epoch,
            ),
            wds.split_by_node,
            wds.split_by_worker,
        ])

        tarfile_to_samples_filter = filters.pipelinefilter(tarfile_to_samples_nothrow)
        pipeline.extend([
            # at this point, we have an iterator over the shards assigned to each worker at each node
            tarfile_to_samples_filter(trie), # tarfile_to_samples_nothrow,  # wds.tarfile_to_samples(handler=log_and_continue),
            wds.shuffle(
                bufsize=_SAMPLE_SHUFFLE_SIZE,
                initial=_SAMPLE_SHUFFLE_INITIAL,
            ),
        ])
    else:
        tarfile_to_samples_filter = filters.pipelinefilter(tarfile_to_samples)
        pipeline.extend([
            wds.split_by_worker,
            # at this point, we have an iterator over the shards assigned to each worker
            tarfile_to_samples_filter(trie) #wds.tarfile_to_samples(handler=log_and_continue),
        ])
    
    pipeline.extend([
            wds.select(filter_no_caption_or_no_image),
            wds.decode("pilrgb", handler=log_and_continue),
    ])
    if return_txt:
        pipeline.extend([
            wds.rename(image="jpg;png", text="txt"),
            wds.map_dict(image=preprocess_img, text=preprocess_txt),
            wds.to_tuple("image", "text", "__path__"),
        ])
    else:
        pipeline.extend([
            wds.rename(image="jpg;png"),
            wds.map_dict(image=preprocess_img),
            wds.to_tuple("image", "__path__"),
        ])
    
    pipeline.extend([
        wds.batched(batch_size, partial=False, collation_fn=torch.utils.data.default_collate),

    ])

    
    dataset = wds.DataPipeline(*pipeline)
    if not resampled:
        assert num_shards >= num_workers * world_size, 'number of shards must be >= total workers' # num_workers <= workers per device
        # roll over and repeat a few samples to get same number of full batches on each node
        round_fn = math.floor if floor else math.ceil
        global_batch_size = batch_size * world_size  # 32*16 = 1024 for bs of 32 and 16 GPUs

        print('global_batch_size: ', global_batch_size)
        print('num_samples: ', num_samples)
        num_batches = round_fn(num_samples / global_batch_size) # num_batches per device = 407,332,084 / 1024   |  Here num_samples = dataset size (407,332,084 for laion400m)
        num_worker_batches = round_fn(num_batches / num_workers)  # number of batches per device per dataloader worker 
        num_batches = num_worker_batches * num_workers # compute num_batches per device again to make it dividable by num_workers
        num_samples = num_batches * global_batch_size # compute num_samples (dataset size) again to make it dividable by num_workers
        dataset = dataset.with_epoch(num_worker_batches)  # each worker is iterating over this to finish one epoch

    else:
        # last batches are partial, eval is done on single (master) node
        num_batches = math.ceil(num_samples / batch_size)

    # import ctypes
    # def worker_init_fn(worker_id):
    #     print(f'✅ init worker {worker_id}')
    #     worker_info = torch.utils.data.get_worker_info()
    #     # worker_info.dataset.trie = ctypes.cast(trie, ctypes.py_object).value
 
    # print('✅ copy trie')
    # import ctypes
    # dataset.trie = ctypes.cast(trie, ctypes.py_object).value
    # worker_info = torch.utils.data.get_worker_info()
    # trie = worker_info.dataset.trie 
    # print('✅✅✅✅ ready trie')
    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=True,
    )

    # add meta-data to dataloader instance for convenience
    dataloader.num_batches = num_batches
    dataloader.num_samples = num_samples


    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def get_captioning400m_data(
    batch_size,
    input_shards,
    preprocess_img,
    epoch=0,
    world_size=1,
    num_workers=1,
    return_txt=True,
    trie=None,
    nb_samples = None,
):

    return get_wds_dataset(
        batch_size,
        input_shards,
        preprocess_img,
        epoch,
        world_size=world_size,
        num_workers=num_workers,
        return_txt=return_txt,
        trie=trie,
        nb_samples = nb_samples
    )
