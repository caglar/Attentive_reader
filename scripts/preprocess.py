import os

import argparse
import cPickle as pkl
import logging

from collections import Counter

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser("Create the dictionary and preprocessing the files.")
parser.add_argument("--source_file_dir", help="The directory of the files for the news located at.")
parser.add_argument("--vocab_type", help="Vocab type has to be either 'q' for questions 'a' for answers or 'd' for description")
parser.add_argument("--out_file_name", help="Output file name")
parser.add_argument("--max_vocab_size", help=("0 to use the whole vocab, otherwise filter top-k"
                                         " according to the uni-gram counts."))
EOS_tokidx = 0

args = parser.parse_args()

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
    if args.vocab_type.lower() == 'q':
        idx = 1
    elif args.vocab_type.lower() == 'a':
        idx = 2
    elif args.vocab_type.lower() == 'd':
        idx = 0

    vocab_counts = Counter()

    for root, dir_, files in os.walk(args.source_file_dir):
        for file_ in files:
            rt_fil = os.path.join(args.source_file_dir, file_)
            source_fil = open(rt_fil, 'r')
            lines = source_fil.readlines()
            data = lines[idx]
            words = data.lower().split(" ")
            words = ["NUM" if w.isdigit() else w for w in words]
            vocab_counts.update(words)
    return vocab_counts


vcounts = retrieve_word_counts()
safe_pickle(vcounts, "%s_vcounts.pkl" % args.vocab_type)


if args.max_vocab_size:
    vcounts = vcounts.most_common(args.max_vocab_size)
else:
    vcounts = list(vcounts)


vocab = {}
vocab['EOS'] = 0
vocab['UNK'] = 1


# A special token for numbers/digits
vocab['NUM'] = 2


for i, vcount in enumerate(vcounts):
    vocab[vcount] = i + 3


logger.info("Saving the vocab.")
safe_pickle(vocab, args.out_file_name)
ivocab = {v: k for k, v in vocab.iteritems()}


logger.info("Saving the ivocab.")
safe_pickle(ivocab, "i_" + args.out_file_name)
