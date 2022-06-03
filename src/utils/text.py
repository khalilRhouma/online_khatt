from functools import reduce
import numpy as np
import unicodedata
import codecs
import re
import csv
import tensorflow as tf
import struct

# Constants
SPACE_TOKEN = "<space>"
SPACE_INDEX = 0
FIRST_INDEX = ord("a") - 1  # 0 is reserved to space


def decodex(txt, mapping):
    out = ""
    for ch in txt:
        out = out + mapping[ch]
    return out


def get_arabic_letters(file_name="Arabic_Mappping.csv"):
    letters_ar = {}
    with open(file_name, encoding="utf-8") as f:
        for line in f:
            (key, val) = line.split(",")
            letters_ar[int(key)] = val.strip()
    letters_ar[83] = " "
    return letters_ar


class Alphabet(object):
    def __init__(self, config_file):
        self._config_file = config_file
        self._label_to_str = {}
        self._str_to_label = {}
        self._size = 0
        if config_file:
            with codecs.open(config_file, "r", "utf-8") as fin:
                for line in fin:
                    if line[0:2] == "\\#":
                        line = "#\n"
                    elif line[0] == "#":
                        continue
                    self._label_to_str[self._size] = line[:-1]  # remove the line ending
                    self._str_to_label[line[:-1]] = self._size
                    self._size += 1

    def _string_from_label(self, label):
        return self._label_to_str[label]

    def _label_from_string(self, string):
        try:
            return self._str_to_label[string]
        except KeyError as e:
            raise KeyError(
                "ERROR: Your transcripts contain characters (e.g. '{}') which do not "
                "occur in data/alphabet.txt! Use  util/check_characters.py to see what "
                "characters are in your [train,dev,test].csv transcripts, and then add "
                "all these to data/alphabet.txt.".format(string)
            ).with_traceback(e.__traceback__)

    def has_char(self, char):
        return char in self._str_to_label

    def encode(self, string):
        res = []
        for char in string:
            res.append(self._label_from_string(char))
        return res

    def decode(self, labels):
        res = ""
        for label in labels:
            res += self._string_from_label(label)
        return res

    def serialize(self):
        # Serialization format is a sequence of (key, value) pairs, where key is
        # a uint16_t and value is a uint16_t length followed by `length` UTF-8
        # encoded bytes with the label.
        res = bytearray()

        # We start by writing the number of pairs in the buffer as uint16_t.
        res += struct.pack("<H", self._size)
        for key, value in self._label_to_str.items():
            value = value.encode("utf-8")
            # struct.pack only takes fixed length strings/buffers, so we have to
            # construct the correct format string with the length of the encoded
            # label.
            res += struct.pack("<HH{}s".format(len(value)), key, len(value), value)
        return bytes(res)

    def size(self):
        return self._size

    def config_file(self):
        return self._config_file


def normalize_txt_file(txt_file, remove_apostrophe=True):
    """
    Given a path to a text file, return contents with unsupported characters removed.
    """
    with codecs.open(txt_file, encoding="utf-8") as open_txt_file:
        return normalize_text(open_txt_file.read(), remove_apostrophe=remove_apostrophe)


def normalize_text(original, remove_apostrophe=True):
    """
    Given a Python string ``original``, remove unsupported characters.

    The only supported characters are letters and apostrophes.
    """
    # convert any unicode characters to ASCII equivalent
    # then ignore anything else and decode to a string
    result = unicodedata.normalize("NFKD", original).encode("ascii", "ignore").decode()
    if remove_apostrophe:
        # remove apostrophes to keep contractions together
        result = result.replace("'", "")
    # return lowercase alphabetic characters and apostrophes (if still present)
    return re.sub("[^a-zA-Z']+", " ", result).strip().lower()


def text_to_char_array(original):
    """
    Given a Python string ``original``, map characters
    to integers and return a numpy array representing the processed string.

    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/text.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """

    # Create list of sentence's words w/spaces replaced by ''
    result = original.replace(" ", "  ")
    result = result.split(" ")

    # Tokenize words into letters adding in SPACE_TOKEN where required
    result = np.hstack([SPACE_TOKEN if xt == "" else list(xt) for xt in result])

    # Return characters mapped into indicies
    return np.asarray(
        [SPACE_INDEX if xt == SPACE_TOKEN else ord(xt) - FIRST_INDEX for xt in result]
    )


def sparse_tuple_from(sequences, dtype=np.int32):
    """
    Create a sparse representention of ``sequences``.

    Args:
        sequences: a list of lists of type dtype where each element is a sequence
    Returns:
        A tuple with (indices, values, shape)

    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/text.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """

    indices = []
    values = []

    for n, seq in enumerate(sequences):
        indices.extend(zip([n] * len(seq), range(len(seq))))
        values.extend(seq)

    indices = np.asarray(indices, dtype=np.int64)
    values = np.asarray(values, dtype=dtype)
    shape = np.asarray([len(sequences), indices.max(0)[1] + 1], dtype=np.int64)

    # return tf.SparseTensor(indices=indices, values=values, shape=shape)
    return indices, values, shape


def sparse_tensor_value_to_texts(value):
    """
    Given a :class:`tf.SparseTensor` ``value``, return an array of Python strings
    representing its values.

    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/text.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """
    return sparse_tuple_to_texts((value.indices, value.values, value.dense_shape))


def sparse_tuple_to_texts(text_tuple):
    """
    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/text.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """

    with open("letters_map.csv") as f:
        # f.readline() # ignore first line (header)
        mydict = dict(csv.reader(f, delimiter=","))

    indices = text_tuple[0]
    values = text_tuple[1]
    results = [""] * text_tuple[2][0]
    for i in range(len(indices)):
        index = indices[i][0]
        c = values[i]
        # c = ' ' if c == SPACE_INDEX else chr(c + FIRST_INDEX)# Fakhr here put ur map
        # whic function that takes number & return char
        # print(mydict[str(c)])
        c = " " if c == SPACE_INDEX else mydict[str(c)] + ","
        results[index] = results[index] + c
    # List of strings
    return results


# """
# def	map_from_index_to_char(indx)
# """
# """
#  define dictioanry that map from letter index into single char for example 0->''
# """


def ndarray_to_text(value):
    """
    This function has been modified from Mozilla DeepSpeech:
    https://github.com/mozilla/DeepSpeech/blob/master/util/text.py

    # This Source Code Form is subject to the terms of the Mozilla Public
    # License, v. 2.0. If a copy of the MPL was not distributed with this
    # file, You can obtain one at http://mozilla.org/MPL/2.0/.
    """
    with open("src/features/utils/letters_map.csv") as f:
        # f.readline() # ignore first line (header)
        mydict = dict(csv.reader(f, delimiter=","))

    results = ""
    for i in range(len(value)):
        # results += chr(value[i] + FIRST_INDEX)
        results += mydict[str(value[i])] + ","
    return results.replace("`", " ")


def gather_nd(params, indices, shape):
    """
    # Function taken from
    # https://github.com/tensorflow/tensorflow/issues/206#issuecomment-229678962

    """
    rank = len(shape)
    flat_params = tf.reshape(params, [-1])
    multipliers = [
        reduce(lambda x, y: x * y, shape[i + 1 :], 1) for i in range(0, rank)
    ]
    indices_unpacked = tf.unstack(
        tf.transpose(indices, [rank - 1] + range(0, rank - 1))
    )
    flat_indices = sum([a * b for a, b in zip(multipliers, indices_unpacked)])
    return tf.gather(flat_params, flat_indices)


def ctc_label_dense_to_sparse(labels, label_lengths, batch_size):
    """
    The CTC implementation in TensorFlow needs labels in a sparse representation,
    but sparse data and queues don't mix well, so we store padded tensors in the
    queue and convert to a sparse representation after dequeuing a batch.

    Taken from
    https://github.com/tensorflow/tensorflow/issues/1742#issuecomment-205291527
    """

    # The second dimension of labels must be equal to the longest label length
    # in the batch
    correct_shape_assert = tf.assert_equal(
        tf.shape(labels)[1], tf.reduce_max(label_lengths)
    )
    with tf.control_dependencies([correct_shape_assert]):
        labels = tf.identity(labels)

    label_shape = tf.shape(labels)
    num_batches_tns = tf.stack([label_shape[0]])
    max_num_labels_tns = tf.stack([label_shape[1]])

    def range_less_than(previous_state, current_input):
        return tf.expand_dims(tf.range(label_shape[1]), 0) < current_input

    init = tf.cast(tf.fill(max_num_labels_tns, 0), tf.bool)
    init = tf.expand_dims(init, 0)
    dense_mask = tf.scan(
        range_less_than, label_lengths, initializer=init, parallel_iterations=1
    )
    dense_mask = dense_mask[:, 0, :]

    label_array = tf.reshape(
        tf.tile(tf.range(0, label_shape[1]), num_batches_tns), label_shape
    )

    label_ind = tf.boolean_mask(label_array, dense_mask)

    batch_array = tf.transpose(
        tf.reshape(
            tf.tile(tf.range(0, label_shape[0]), max_num_labels_tns),
            tf.reverse(label_shape, [0]),
        )
    )
    batch_ind = tf.boolean_mask(batch_array, dense_mask)
    batch_label = tf.concat([batch_ind, label_ind], 0)
    indices = tf.transpose(tf.reshape(batch_label, [2, -1]))
    shape = [batch_size, tf.reduce_max(label_lengths)]
    vals_sparse = gather_nd(labels, indices, shape)

    return tf.SparseTensor(tf.to_int64(indices), vals_sparse, tf.to_int64(label_shape))
