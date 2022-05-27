import os
from math import ceil
from random import random
from glob import glob
from configparser import ConfigParser
import logging
from collections import namedtuple

from src.utils.load_hw_to_mem import get_handwriting_and_transcript, pad_sequences
from src.utils.text import sparse_tuple_from
from src.utils.set_dirs import get_data_dir

DataSets = namedtuple("DataSets", "train dev test")

import numpy as np


def handwriting_to_input_vector(points, numcep, numcontext):  # modify mozilla function
    """
    Turn an audio file into feature representation.

    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/audio.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """

    # Get Features

    orig_inputs = points  # (57,20)

    # We only keep every second feature (BiRNN stride = 2)
    orig_inputs = orig_inputs[::2]

    # For each time slice of the training set, we need to copy the context this makes
    # the numcep dimensions vector into a numcep + 2*numcep*numcontext dimensions
    # because of:
    #  - numcep dimensions for the current point feature set
    #  - numcontext*numcep dimensions for each of the past and future (x2) point feature set
    # => so numcep + 2*numcontext*numcep
    train_inputs = np.array([], np.float32)
    train_inputs.resize((orig_inputs.shape[0], numcep + 2 * numcep * numcontext))

    # Prepare pre-fix post fix context
    empty_point = np.array([])
    empty_point.resize((numcep))

    # Prepare train_inputs with past and future contexts
    time_slices = range(train_inputs.shape[0])
    context_past_min = time_slices[0] + numcontext
    context_future_max = time_slices[-1] - numcontext
    for time_slice in time_slices:
        # Reminder: array[start:stop:step]
        # slices from indice |start| up to |stop| (not included), every |step|

        # Add empty context data of the correct size to the start and end
        # of the point feature matrix

        # Pick up to numcontext time slices in the past, and complete with empty
        # point features
        need_empty_past = max(0, (context_past_min - time_slice))
        empty_source_past = list(empty_point for empty_slots in range(need_empty_past))
        data_source_past = orig_inputs[max(0, time_slice - numcontext) : time_slice]
        assert len(empty_source_past) + len(data_source_past) == numcontext

        # Pick up to numcontext time slices in the future, and complete with empty
        # point features
        need_empty_future = max(0, (time_slice - context_future_max))
        empty_source_future = list(
            empty_point for empty_slots in range(need_empty_future)
        )
        data_source_future = orig_inputs[time_slice + 1 : time_slice + numcontext + 1]
        assert len(empty_source_future) + len(data_source_future) == numcontext

        if need_empty_past:
            past = np.concatenate((empty_source_past, data_source_past))
        else:
            past = data_source_past

        if need_empty_future:
            future = np.concatenate((data_source_future, empty_source_future))
        else:
            future = data_source_future

        past = np.reshape(past, numcontext * numcep)
        now = orig_inputs[time_slice]
        future = np.reshape(future, numcontext * numcep)

        train_inputs[time_slice] = np.concatenate((past, now, future))
        assert len(train_inputs[time_slice]) == numcep + 2 * numcep * numcontext

    # Scale/standardize the inputs
    # This can be done more efficiently in the TensorFlow graph
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    return train_inputs


def pad_sequences(
    sequences,
    maxlen=None,
    dtype=np.float32,
    padding="post",
    truncating="post",
    value=0.0,
):

    """
    # From TensorLayer:
    # http://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/prepro.html

    Pads each sequence to the same length of the longest sequence.

        If maxlen is provided, any sequence longer than maxlen is truncated to
        maxlen. Truncation happens off either the beginning or the end
        (default) of the sequence. Supports post-padding (default) and
        pre-padding.

        Args:
            sequences: list of lists where each element is a sequence
            maxlen: int, maximum length
            dtype: type to cast the resulting sequence.
            padding: 'pre' or 'post', pad either before or after each sequence.
            truncating: 'pre' or 'post', remove values from sequences larger
            than maxlen either in the beginning or in the end of the sequence
            value: float, value to pad the sequences to the desired value.

        Returns:
            numpy.ndarray: Padded sequences shape = (number_of_sequences, maxlen)
            numpy.ndarray: original sequence lengths
    """
    lengths = np.asarray([len(s) for s in sequences], dtype=np.int64)

    nb_samples = len(sequences)
    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = ()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    x = (np.ones((nb_samples, maxlen) + sample_shape) * value).astype(dtype)
    for idx, s in enumerate(sequences):
        if len(s) == 0:
            continue  # empty list was found
        if truncating == "pre":
            trunc = s[-maxlen:]
        elif truncating == "post":
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError(
                "Shape of sample %s of sequence at position %s is different from "
                "expected shape %s" % (trunc.shape[1:], idx, sample_shape)
            )

        if padding == "post":
            x[idx, : len(trunc)] = trunc
        elif padding == "pre":
            x[idx, -len(trunc) :] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x, lengths


def read_datasets(conf_path, sets, numcep, numcontext, thread_count=8):
    """Main function to create DataSet objects.

    This function calls an internal function _get_data_set_dict that
    reads the configuration file. Then it calls the internal function _read_data_set
    which collects the text files in the data directories, returning a DataSet object.
    This function returns a DataSets object containing the requested datasets.

    Args:
        sets (list):   List of datasets to create. Options are: 'train', 'dev', 'test'
        numcep (int):  Number of features to compute.
        numcontext (int): For each time point, number of contextual samples to include.
        thread_count (int): Number of threads

    Returns:
        DataSets: A single `DataSets` instance containing each of the requested datasets

        E.g., when sets=['train'], datasets.train exists, with methods to retrieve
        examples.

    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/importers/librivox.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """
    data_dir, dataset_config = _get_data_set_dict(conf_path, sets)

    def _read_data_set(config):
        path = os.path.join(data_dir, config["dir_pattern"])
        return DataSet.from_directory(
            path,
            thread_count=thread_count,
            batch_size=config["batch_size"],
            numcep=numcep,
            numcontext=numcontext,
            start_idx=config["start_idx"],
            limit=config["limit"],
            sort=config["sort"],
        )

    datasets = {
        name: _read_data_set(dataset_config[name]) if name in sets else None
        for name in ("train", "dev", "test")
    }
    return DataSets(**datasets)


class DataSet:
    """
    Train/test/dev dataset API for loading via threads and delivering batches.

    This class has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/importers/librivox.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """

    def __init__(self, txt_files, thread_count, batch_size, numcep, numcontext):
        self._coord = None
        self._numcep = numcep
        self._txt_files = txt_files
        self._batch_size = batch_size
        self._numcontext = numcontext
        self._start_idx = 0

    @classmethod
    def from_directory(
        cls,
        dirpath,
        thread_count,
        batch_size,
        numcep,
        numcontext,
        start_idx=0,
        limit=0,
        sort=None,
    ):
        if not os.path.exists(dirpath):
            raise IOError("'%s' does not exist" % dirpath)
        txt_files = txt_filenames(dirpath, start_idx=start_idx, limit=limit, sort=sort)
        if len(txt_files) == 0:
            raise RuntimeError(
                "start_idx=%d and limit=%d arguments result in zero files"
                % (start_idx, limit)
            )
        return cls(txt_files, thread_count, batch_size, numcep, numcontext)

    def next_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self._batch_size

        end_idx = min(len(self._txt_files), self._start_idx + batch_size)
        idx_list = range(self._start_idx, end_idx)
        txt_files = [self._txt_files[i] for i in idx_list]
        # Fakhr Here
        hw_files = [x.replace("_target.npy", "_input.npy") for x in txt_files]

        # write the files paths to file
        with open("files_paths.txt", "a") as file:
            for item in hw_files:
                file.write("%s" % item)
                file.write("\n")
                # print(hw_files)

        (source, _, target, _) = get_handwriting_and_transcript(
            txt_files, hw_files, self._numcep, self._numcontext
        )
        self._start_idx += batch_size
        # Verify that the start_idx is not larger than total available sample size
        if self._start_idx >= self.size:
            self._start_idx = 0

        # Pad input to max_time_step of this batch
        source, source_lengths = pad_sequences(source)
        sparse_labels = sparse_tuple_from(target)
        return source, source_lengths, sparse_labels

    @property
    def files(self):
        return self._txt_files

    @property
    def size(self):
        return len(self.files)

    @property
    def total_batches(self):
        # Note: If len(_txt_files) % _batch_size != 0, this re-uses initial _txt_files
        return int(ceil(float(len(self._txt_files)) / float(self._batch_size)))


# END DataSet

SORTS = ["filesize_low_high", "filesize_high_low", "alpha", "random"]


def txt_filenames(dataset_path, start_idx=0, limit=None, sort="alpha"):
    # Obtain list of txt files
    # txt_files = glob(os.path.join(dataset_path, "*.txt"))
    txt_files = glob(os.path.join(dataset_path, "*_target.npy"))
    limit = limit or len(txt_files)

    # Optional: sort files to improve padding performance
    if sort not in SORTS:
        raise ValueError("sort must be one of [%s]", SORTS)
    reverse = False
    key = None
    if "filesize" in sort:
        key = os.path.getsize
    if sort == "filesize_high_low":
        reverse = True
    elif sort == "random":
        key = lambda *args: random()
    txt_files = sorted(txt_files, key=key, reverse=reverse)

    return txt_files[start_idx : limit + start_idx]


def _get_data_set_dict(conf_path, sets):
    parser = ConfigParser(os.environ)
    parser.read(conf_path)
    config_header = "data"
    data_dir = get_data_dir(parser.get(config_header, "data_dir"))
    data_dict = {}

    if "train" in sets:
        d = {}
        d["dir_pattern"] = parser.get(config_header, "dir_pattern_train")
        d["limit"] = parser.getint(config_header, "n_train_limit")
        d["sort"] = parser.get(config_header, "sort_train")
        d["batch_size"] = parser.getint(config_header, "batch_size_train")
        d["start_idx"] = parser.getint(config_header, "start_idx_init_train")
        data_dict["train"] = d
        logging.debug("Training configuration: %s", str(d))

    if "dev" in sets:
        d = {}
        d["dir_pattern"] = parser.get(config_header, "dir_pattern_dev")
        d["limit"] = parser.getint(config_header, "n_dev_limit")
        d["sort"] = parser.get(config_header, "sort_dev")
        d["batch_size"] = parser.getint(config_header, "batch_size_dev")
        d["start_idx"] = parser.getint(config_header, "start_idx_init_dev")
        data_dict["dev"] = d
        logging.debug("Dev configuration: %s", str(d))

    if "test" in sets:
        d = {}
        d["dir_pattern"] = parser.get(config_header, "dir_pattern_test")
        d["limit"] = parser.getint(config_header, "n_test_limit")
        d["sort"] = parser.get(config_header, "sort_test")
        d["batch_size"] = parser.getint(config_header, "batch_size_test")
        d["start_idx"] = parser.getint(config_header, "start_idx_init_test")
        data_dict["test"] = d
        logging.debug("Test configuration: %s", str(d))

    return data_dir, data_dict
