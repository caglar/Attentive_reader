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

parser.add_argument("--vocab_desc",
                    type=argparse.FileType('r'),
                    help="Vocabulary file for description")

parser.add_argument("--vocab_ans",
                    type=argparse.FileType('r'),
                    help="Vocabulary file for answer")

parser.add_argument("--vocab_q",
                    type=argparse.FileType('r'),
                    help="Vocabulary file for question")

parser.add_argument("--output_file",
                    type=argparse.FileType("w"),
                    help="Output HDF5 file")

args = parser.parse_args()

q_vocab = pkl.load(args.vocab_q)
ans_vocab = pkl.load(args.vocab_ans)
desc_vocab = pkl.load(args.vocab_desc)


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

def get_sentence_rep(descs):
    sents = []
    wordsinsents = []
    for i, dw in  enumerate(descs):
        tok = desc_vocab["NUM"] if dw.isdigit() else desc_vocab[dw] if dw in desc_vocab \
            else desc_vocab["UNK"]
        if tok == desc_vocab["."] or tok == desc_vocab["!"] or tok == desc_vocab["?"]:
            wordsinsents.append(tok)
            if len(descs) - 1 == i:
                wordsinsents.append(0)
            sents.append(np.array(wordsinsents, dtype="uint32"))
            wordsinsents = []
        else:
            wordsinsents.append(tok)
            if len(descs) - 1 == i:
                wordsinsents.append(desc_vocab['EOS'])
    return sents

def get_triple_indices(descs, qs, ans):
    d_idxs =get_sentence_rep(descs)
    ans_idxs = np.array([ans_vocab["NUM"] if a.isdigit() else ans_vocab[a] if a in ans_vocab \
            else ans_vocab["UNK"] for a in ans], dtype="uint32")
    q_idxs = np.array([q_vocab["NUM"] if q.isdigit() else q_vocab[q] if q in q_vocab \
            else q_vocab["UNK"] for q in qs] + [0], dtype="uint32")
    return d_idxs, ans_idxs, q_idxs

print "Started walking on the files."

for root, dir_, files in os.walk(args.input_dir):
    for file_ in files:
        import ipdb; ipdb.set_trace()
        fil_ = os.path.join(args.input_dir, file_)
        source_fil = open(fil_, 'r')
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
