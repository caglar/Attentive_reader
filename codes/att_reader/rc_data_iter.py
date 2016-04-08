import threading
import Queue
import logging

import numpy
import tables
from six import Iterator
import sys
logger = logging.getLogger(__name__)

numpy.random.seed(123)


def prepare_data(seqs_x, maxlen=None, n_words=30000):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]

    if maxlen is not None:
        new_seqs_x = []
        new_lengths_x = []
        for l_x, s_x in zip(lengths_x, seqs_x):
            if l_x < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)

        lengths_x = new_lengths_x
        seqs_x = new_seqs_x

        if len(lengths_x) < 1:
            return None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('uint32')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')

    for idx, s_x in enumerate(seqs_x):
        s_x[numpy.where(s_x >= n_words-1)] = 1
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.

    return x, x_mask


def parse_description_sents(seqs_x, sent_end_toks, eos_tok_id, quote_tok_id):
    batch = []
    words = []
    sents = []
    for i, desc in enumerate(seqs_x):
        words = []
        sents = []
        quote = False

        for j, wval in enumerate(desc):
            if j != len(desc) - 1 and \
                    desc[j+1] == quote_tok_id and quote:
                words.append(wval)
            elif wval in sent_end_toks:
                if j != len(desc) - 2 and not quote:
                    words.append(wval)
                    sents.append(words)
                    words = []
                    quote = False
                elif desc[j+1] == eos_tok_id:
                    words.append(wval)
                    sents.append(words)
                else:
                    words.append(wval)
                    sents.append(words)
                    words = []
                    quote = False
            else:
                words.append(wval)

            if desc[j] == quote_tok_id:
                quote = not quote

        batch.append(sents)

    return batch


def load_data(path=None,
              valid_path=None,
              test_path=None,
              batch_size=128,
              **kwargs):
    '''
       Loads the dataset.
    '''
    if path is None:
        path = '/data/lisatmp4/gulcehrc/reading_comprehension_data/cleaned_cnn/cnn_training_data.h5'

    print "Using training data ", path
    #############
    # LOAD DATA #
    #############

    print '... initializing data iterators'
    train = PytablesRCDataIterator(batch_size,
                                   path,
                                   use_infinite_loop=False,
                                   **kwargs)

    valid = PytablesRCDataIterator(batch_size,
                                   valid_path,
                                   shuffle=False,
                                   use_infinite_loop=False,
                                   **kwargs) if valid_path else None

    test = PytablesRCDataIterator(batch_size,
                                  test_path,
                                  shuffle=False,
                                  use_infinite_loop=False,
                                  **kwargs) if test_path else None

    return train, valid, test


def get_length(path):
    if tables.__version__[0] == '2':
        target_table = tables.openFile(path, 'r')
        target_index = target_table.getNode('/indices')
    else:
        target_table = tables.open_file(path, 'r')
        target_index = target_table.get_node('/indices')

    return target_index.shape[0]


def synchronized_open_file(*args, **kwargs):
    if tables.__version__[0] == '2':
        tbf = tables.openFile(*args, **kwargs)
    else:
        tbf = tables.open_file(*args, **kwargs)
    return tbf


class PytablesRCDataIterator(Iterator):

    def __init__(self,
                 batch_size,
                 target_file=None,
                 dtype="uint32",
                 dtable_name='/desc_data',
                 dindex_name='/desc_indices',
                 qtable_name='/q_data',
                 qindex_name='/q_indices',
                 atable_name='/ans_data',
                 aindex_name='/ans_indices',
                 can_fit=False,
                 start=0,
                 stop=-1,
                 use_sentence_reps=False,
                 n_batches_in_bucket=12,
                 sent_end_tok_ids=None,
                 quote_tok_id=None,
                 eos_tok_id=None,
                 queue_size=1000,
                 cache_size=1000,
                 shuffle=True,
                 use_infinite_loop=False):

        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        self.queue = None
        self.gather = None
        self.started = False
        self.use_sentence_reps = use_sentence_reps
        self.sent_end_tok_ids = sent_end_tok_ids

        self.source_buffer = []
        self.target_buffer = []
        self.question_buffer = []

        self.n_batches_in_bucket = n_batches_in_bucket
        self.max_kbatches = batch_size * n_batches_in_bucket

        self.eos_tok_id = eos_tok_id
        self.offset = self.start
        self.quote_tok_id = quote_tok_id
        self.exit_flag = False
        self.load_rc_data()
        self.done = True

    def load_rc_data(self):
        driver = None
        if self.can_fit:
            driver = "H5FD_CORE"

        self.started = True
        target_table = synchronized_open_file(self.target_file, 'r',
                                              driver=driver)
        print "Opened the file."

        if tables.__version__[0] == '2':
            self.d_data, self.d_index = (target_table.getNode(self.dtable_name),
                target_table.getNode(self.dindex_name))
            self.q_data, self.q_index = (target_table.getNode(self.qtable_name),
                target_table.getNode(self.qindex_name))
            self.a_data, self.a_index = (target_table.getNode(self.atable_name),
                target_table.getNode(self.aindex_name))
        else:
            self.d_data, self.d_index = (target_table.get_node(self.dtable_name),
                target_table.get_node(self.dindex_name))
            self.q_data, self.q_index = (target_table.get_node(self.qtable_name),
                target_table.get_node(self.qindex_name))
            self.a_data, self.a_index = (target_table.get_node(self.atable_name),
                target_table.get_node(self.aindex_name))

        self.data_len = self.d_index.shape[0]
        if self.stop <= 0:
            self.stop = self.data_len

    def __iter__(self):
        return self

    def next(self):
        max_qlen = 0
        max_dlen = 0

        if self.offset == -1:
            self.offset = 0
            self.start = self.offset
            if self.shuffle:
                self.offset = numpy.random.randint(low=self.start, high=self.stop)

        desc_ngrams = []
        q_ngrams = []
        ans = []

        assert len(self.source_buffer) == len(self.target_buffer), 'Buffer size mismatch!'
        assert len(self.source_buffer) == len(self.question_buffer), 'Buffer size mismatch!'

        if len(self.source_buffer) == 0:
            dlens = []
            while len(self.source_buffer) < self.max_kbatches:
                dlen, dpos = self.d_index[self.offset]['length'], \
                        self.d_index[self.offset]['pos']
                qlen, qpos = self.q_index[self.offset]['length'], \
                        self.q_index[self.offset]['pos']
                apos = self.a_index[self.offset]['pos']
                if dlen > max_dlen:
                    max_dlen = dlen

                if qlen > max_qlen:
                    max_qlen = qlen
                dlens.append(dlen)
                self.offset += 1
                if self.offset >= self.data_len or self.offset >= self.stop:
                    if self.use_infinite_loop:
                        self.offset = self.start
                    else:
                        self.done = True
                        raise StopIteration

                self.source_buffer.append(self.d_data[dpos:dpos+dlen])
                self.question_buffer.append(self.q_data[qpos:qpos+qlen])
                self.target_buffer.append(self.a_data[apos])

            didxs = numpy.array(dlens).argsort()
            _sbuf = [self.source_buffer[i] for i in didxs]
            _tbuf = [self.target_buffer[i] for i in didxs]
            _qbuf = [self.question_buffer[i] for i in didxs]

            self.source_buffer = _sbuf
            self.target_buffer = _tbuf
            self.question_buffer = _qbuf

        while True:
            try:
                ss = self.source_buffer.pop()
                ts = self.target_buffer.pop()
                qs = self.question_buffer.pop()
            except IndexError:
                break

            desc_ngrams.append(ss)
            q_ngrams.append(qs)
            ans.append(ts)

            if len(desc_ngrams) >= self.batch_size or \
                    len(q_ngrams) >= self.batch_size or \
                    len(ans) >= self.batch_size:
                        break

        if len(desc_ngrams) <= 0 or len(q_ngrams) <= 0 or \
                len(ans) <= 0:
                    raise StopIteration

        if self.use_sentence_reps:
            desc_ngrams = parse_description_sents(desc_ngrams,
                                                  self.sent_end_tok_ids,
                                                  self.eos_tok_id,
                                                  self.quote_tok_id)

        return desc_ngrams, q_ngrams, ans, max_dlen, max_qlen

    def reset(self):
        assert self.started, "You should start the iterator first!"
        self.offset = self.start
        self.done = False
