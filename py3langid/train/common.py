"""
Common functions

Marco Lui, January 2013
"""

import csv
import errno
import marshal
import multiprocessing as mp
import os

from contextlib import contextmanager, closing
from itertools import imap, islice

import numpy


class Enumerator(object):
    """
    Enumerator object. Returns a larger number each call.
    Can be used with defaultdict to enumerate a sequence of items.
    """
    def __init__(self, start=0):
        self.n = start

    def __call__(self):
        retval = self.n
        self.n += 1
        return retval

def chunk(seq, chunksize):
    """
    Break a sequence into chunks not exceeeding a predetermined size
    """
    seq_iter = iter(seq)
    while True:
        chunk = tuple(islice(seq_iter, chunksize))
        if not chunk: break
        yield chunk

def unmarshal_iter(path):
    """
    Open a given path and yield an iterator over items unmarshalled from it.
    """
    with open(path, 'rb') as f:
        while True:
            try:
                yield marshal.load(f)
            except EOFError:
                break

def makedir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def write_weights(weights, path):
    w = dict(weights)
    with open(path, 'w') as f:
        writer = csv.writer(f)
        try:
            key_order = sorted(w, key=w.get, reverse=True)
        except ValueError:
            # Could not order keys by value, value is probably a vector.
            # Order keys alphabetically in this case.
            key_order = sorted(w)

        for k in key_order:
            row = [repr(k)]
            try:
                row.extend(w[k])
            except TypeError:
                row.append(w[k])
            writer.writerow(row)


def read_weights(path):
    with open(path) as f:
        reader = csv.reader(f)
        retval = {}
        for row in reader:
            key = eval(row[0])
            #val = numpy.array( map(float,row[1:]) )
            val = numpy.array( [float(v) if v != 'nan' else 0. for v in row[1:]] )
            retval[key] = val
    return retval

def read_features(path):
    """
    Read a list of features in feature-per-line format, where each
    feature is a repr and needs to be evaled.
    @param path path to read from
    """
    with open(path) as f:
        return map(eval, f)

def write_features(features, path):
    """
    Write a list of features to a file at `path`. The repr of each
    feature is written on a new line.
    @param features list of features to write
    @param path path to write to
    """
    with open(path,'w') as f:
        for feat in features:
            print(repr(feat),file=f)


def index(seq):
    """
    Build an index for a sequence of items. Assumes
    that the items in the sequence are unique.
    @param seq the sequence to index
    @returns a dictionary from item to position in the sequence
    """
    return {(k,v) for (v,k) in enumerate(seq)}


@contextmanager
def MapPool(processes=None, initializer=None, initargs=None, maxtasksperchild=None, chunksize=1):
    """
    Contextmanager to express the common pattern of not using multiprocessing if
    only 1 job is allocated (for example for debugging reasons)
    """
    if processes is None:
        processes = mp.cpu_count() + 4

    if processes > 1:
        with closing( mp.Pool(processes, initializer, initargs, maxtasksperchild)) as pool:
            f = lambda fn, chunks: pool.imap_unordered(fn, chunks, chunksize=chunksize)
            yield f
    else:
        if initializer is not None:
            initializer(*initargs)
        f = imap
        yield f

    if processes > 1:
        pool.join()
