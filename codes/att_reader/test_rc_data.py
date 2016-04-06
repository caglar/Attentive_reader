from rc_data_iter import PytablesRCDataIterator, load_data
import os
import cPickle as pkl

vdir = "/data/lisatmp4/gulcehrc/reading_comprehension_data/cleaned_cnn/"
vocab = pkl.load(open(os.path.join(vdir, "i_cleaned_cnn_vocab.pkl")))


def from_ids_to_sent(xs, vocab):
    words = []
    for x in xs:
        words.append(vocab[x])
    return " ".join(words)

test_path = ""
train, _, _ = load_data()

for batch in train:
    desc = batch[0]
    q = batch[1]
    a = batch[2]
    for d_, q_, a_ in zip(desc, q, a):
        print "DESC:"
        print from_ids_to_sent(d_, vocab)
        print "Q:"
        print from_ids_to_sent(q_, vocab)
        print "A:"
        print vocab[a_]
        raw_input()
        print "----------------------------------------------"
