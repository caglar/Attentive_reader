import os

import argparse
import cPickle as pkl
import logging
from tqdm import tqdm

from collections import Counter, OrderedDict

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


def validate_word(w):
    if w.isdigit():
        return "NUM"
    elif w!='' and w is not None:
        return w
    return -1


"""
In the cleaned RC-data:
    * 1st line is the description 'd'.
    * 2nd line is the question 'q'.
    * 3rd line is the answer 'a'.
"""
def retrieve_word_counts():
    """
    if args.vocab_type.lower() == 'q':
        idx = 1
    elif args.vocab_type.lower() == 'a':
        idx = 2
    elif args.vocab_type.lower() == 'd':
        idx = 0
    """
    vocab_counts = Counter()
    for root, dir_, files in os.walk(args.source_file_dir):
        nfiles = len(files)
        for fidx in tqdm(xrange(nfiles)):
            file_ = files[fidx]
            rt_fil = os.path.join(args.source_file_dir, file_)
            source_fil = open(rt_fil, 'r')
            lines = source_fil.readlines()
            for data in lines:
                words = data.lower().split(" ")
                pruned_words = []
                for w in words:
                    nw = validate_word(w)
                    if nw >= -1:
                        pruned_words.append(nw)
                #words = ["NUM" if w.isdigit() else w if (w != '' and w is not None) for w in words]
                vocab_counts.update(pruned_words)
    return vocab_counts


vcounts = retrieve_word_counts()
safe_pickle(vcounts, "%s_vcounts.pkl" % args.vocab_type)

if args.max_vocab_size:
    vcounts_ = vcounts.most_common(args.max_vocab_size)
else:
    vcounts_ = list(vcounts)

vocab = OrderedDict({})

vocab['NIL'] = 0
vocab['EOS'] = 1
vocab['UNK'] = 2
# A special token for numbers/digits
vocab['NUM'] = 3

i=4
for vcount in vcounts_:
    if vcounts[vcount] > 1:
        print vcount, i
        vocab[vcount] = i
        i += 1

logger.info("Saving the vocab.")
safe_pickle(vocab, args.out_file_name)
ivocab = {v: k for k, v in vocab.iteritems()}


# logger.info("Saving the ivocab.")
# safe_pickle(ivocab, args.out_file_name + "_i")
