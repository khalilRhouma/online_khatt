#!/usr/bin/env python

import sys
import argparse
import scipy.cluster.vq as vq
import numpy as np
import glob, os
from pathlib import Path
from src.features.preprocessing import preprocess_handwriting
from src.features.feature import calculate_feature_vector_sequence

np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

# NORM_ARGS = ["smooth", "slope", "resample", "slant", "height", "origin"]
NORM_ARGS = ["origin", "filp_h", "smooth", "slope", "resample", "slant", "height"]
FEAT_ARGS = [
    "x_cor",
    "y_cor",
    "penup",
    "dir",
    "curv",
    "vic_aspect",
    "vic_curl",
    "vic_line",
    "vic_slope",
    "bitmap",
]


def read_handwriting_from_file(file):
    """
    Reads points of an online handwriting from a textfile where
    each line is formatted as "x y penup". All entries have to be integers.
    penup is 0/1 depending on the state of the pen. An optional annotation of
    the presented word can be given on the first line

    :param file: Textfile containing points of a handwriting.
    :return: Numpy array with columns x, y and penup, annotation or None
    """
    points = []
    annotation = None
    with open(file, "r") as ink_file:
        for line in ink_file:
            parts = line.split(" ")
            if len(parts) != 3:
                annotation = line.strip()
                continue
            points.append([float(p.strip()) for p in parts])
    return np.array(points), annotation


def process_single_file(file, filename, outfilename, normalize=True):
    print("Importing {}...".format(filename))
    ink, word = read_handwriting_from_file(filename)
    print("Read {} points for word '{}'.".format(len(ink), word))
    if len(ink) < 3:
        file.close()
        os.rename(filename, "Error/" + filename)
        return
    # ink[:,2]=np.abs((ink[:,2]-1)*-1)
    # print(ink)

    if normalize:
        print("Normalizing handwriting...")
        ink = preprocess_handwriting(ink, NORM_ARGS)
        # np.savetxt(outfilename.replace("npy",".txt"),ink)

    print("Calculating feature vector sequence...")
    feat_seq_mat = calculate_feature_vector_sequence(ink, FEAT_ARGS)

    # otherwise we just save the feature vector sequence
    print("Writing {}...".format(outfilename))
    feat_seq_mat = feat_seq_mat.astype("float32")
    np.save(outfilename, feat_seq_mat)
    # feat_seq_mat.tofile(outfilename)


def main():

    # os.chdir("features/")
    ink_paths = list(Path("features/inks").glob("*.txt"))
    save_dir = Path("features/data")
    print(len(ink_paths))
    for filename in ink_paths:
        with open(filename) as file:
            process_single_file(
                file, filename, f"{save_dir}/{filename.stem}_input.npy", normalize=True
            )
            file.close()
            try:
                os.rename(filename, "processed/" + filename)
            except Exception as e:
                pass


if __name__ == "__main__":
    main()
