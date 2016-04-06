import os

import argparse
import cPickle as pkl
import logging

from collections import Counter

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("Create the dictionary and preprocessing the files.")
parser.add_argument("--source_files_dir", help="The directory of the files for the news located at.", default=".")
parser.add_argument("--out_file_name", help="Output file name", required=True)
parser.add_argument("--max_vocab_size", help=("0 to use the whole vocab, otherwise filter top-k "
                                              " according to the unigram counts."))
parser.add_argument("--file_prfx", help="file prefix name")

EOS_tokidx = 0

args = parser.parse_args()

sfxs = ["a", "p", "q"]
file_names = ["%s%s" % (args.file_prfx, sfx) for sfx in sfxs]

assert args.out_file_name is not None


def safe_pickle(obj, filename):
    if os.path.isfile(filename):
        logger.info("Overwriting %s." % filename)
    else:
        logger.info("Saving to %s." % filename)
    with open(filename, 'wb') as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)


"""
In the cleaned RC-data:
    * 1st line is the description 'd'.
    * 2nd line is the question 'q'.
    * 3rd line is the answer 'a'.
"""
def retrieve_word_counts():
    vocab_counts = Counter()

    for file_ in file_names:
        rt_fil = os.path.join(args.source_files_dir, file_)
        source_fil = open(rt_fil, 'r')
        lines = source_fil.readlines()
        for line in lines:
            words = line.lower().split(" ")
            nwords = []
            for w in words:
                w = w.strip()
                if w.isdigit():
                    nwords.append(w)
                elif w != '' and w is not None:
                    nwords.append(w)
            vocab_counts.update(nwords)
    return vocab_counts

vcounts = retrieve_word_counts()
safe_pickle(vcounts, "%s_vcounts.pkl" % ("all"))

if args.max_vocab_size:
    vcounts_ = vcounts.most_common(args.max_vocab_size)
else:
    vcounts_ = list(vcounts)

vocab = {}
vocab['NIL'] = 0
vocab['EOS'] = 1
vocab['UNK'] = 2

# A special token for numbers/digits
vocab['NUM'] = 3

i = 4
for vcount in vcounts_:
    if vcounts[vcount] > 1:
        print vcount, i
        vocab[vcount] = i
        i += 1


logger.info("Saving the vocab.")
safe_pickle(vocab, args.out_file_name)
ivocab = {v: k for k, v in vocab.iteritems()}


logger.info("Saving the ivocab.")
safe_pickle(ivocab, "i_" + args.out_file_name)
