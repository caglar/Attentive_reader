#!/usr/bin/env python

import argparse
import cPickle as pkl

import sys

import os

import tables
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("--input_dir",
                    help="Input directory for the files")

parser.add_argument("--vocab",
                    type=argparse.FileType('r'),
                    help="Vocabulary file for description")

parser.add_argument("--output_file",
                    type=argparse.FileType("w"),
                    help="Output HDF5 file")

args = parser.parse_args()

vocab = pkl.load(args.vocab)


class Index(tables.IsDescription):
    pos = tables.UInt32Col()
    length = tables.UInt32Col()


class SingletonIndex(tables.IsDescription):
    pos = tables.UInt32Col()


f = args.output_file
f = tables.open_file(f.name, f.mode)

q_earrays = f.createEArray(f.root, 'q_data',
    tables.Int32Atom(), shape=(0,))

ans_earrays = f.createEArray(f.root, 'ans_data',
    tables.Int32Atom(), shape=(0,))

desc_earrays = f.createEArray(f.root, 'desc_data',
    tables.Int32Atom(), shape=(0,))


indices = f.createTable("/", 'desc_indices',
    Index, "a table of indices for descs and lengths")

qs_ids = f.createTable("/", 'q_indices',
    Index, "a table of indices qs and lengths")

ans_ids = f.createTable("/", 'ans_indices',
    SingletonIndex, "a table of indices for answer")

count = 0
pos_desc = 0
pos_q = 0
pos_a = 0


def get_triple_indices(descs, qs, ans):
    d_idxs = np.array([vocab["NUM"] if d.isdigit() else vocab[d] if d in vocab \
            else vocab["UNK"] for d in descs] + [vocab['EOS']], dtype="uint32")
    ans_idxs = np.array([vocab["NUM"] if a.isdigit() else vocab[a] if a in vocab \
            else vocab["UNK"] for a in ans], dtype="uint32")
    q_idxs = np.array([vocab["NUM"] if q.isdigit() else vocab[q] if q in vocab \
            else vocab["UNK"] for q in qs] + [vocab['EOS']], dtype="uint32")
    return d_idxs, ans_idxs, q_idxs


print "Started walking on the files."
for root, dir_, files in os.walk(args.input_dir):
    for file_ in files:
        fil_ = os.path.join(args.input_dir, file_)
        with open(fil_, 'r') as source_fil:
            data = source_fil.readlines()

        desc = data[0].split(" ")
        q = data[1].split(" ")
        ans = data[2].split(" ")
        d_ids, a_ids, q_ids = get_triple_indices(desc, q, ans)
        q_earrays.append(q_ids)
        ans_earrays.append(a_ids)
        desc_earrays.append(d_ids)
        d_ind = indices.row
        d_ind['pos'] = pos_desc
        d_ind['length'] = len(d_ids)
        d_ind.append()
        q_ind = qs_ids.row
        q_ind['pos'] = pos_q
        q_ind['length'] = len(q_ids)
        q_ind.append()
        a_ind = ans_ids.row
        a_ind['pos'] = pos_a
        a_ind.append()
        pos_desc += len(d_ids)
        pos_q += len(q_ids)
        pos_a += len(a_ids)

        count += 1

        if count % 100000 == 0:
            print count,
            sys.stdout.flush()
            indices.flush()
            qs_ids.flush()
            ans_ids.flush()
        elif count % 10000 == 0:
            print '.',
            sys.stdout.flush()

f.close()

print 'processed', count, 'documents'
