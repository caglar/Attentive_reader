import os


import cPickle as pkl
from six import Iterator
import numpy
import logging

from utils import get_entities, create_mappings, create_mask_entities


numpy.random.seed(1234)


def get_data_files(mode="top8"):
    if mode == "top8":
        vdir = "/u/yyu/stor/cnn/pureNoUnify/att8/"
        vpath = os.path.join(vdir, "dict_v8.pkl")

        train_files = [os.path.join(vdir, "train8v2_pass.pkl"),
                    os.path.join(vdir, "train8v2_qs.pkl"),
                    os.path.join(vdir, "train8v2_ans.pkl")]

        valid_files = [os.path.join(vdir, "valid8v2_pass.pkl"),
                    os.path.join(vdir, "valid8v2_qs.pkl"),
                    os.path.join(vdir, "valid8v2_ans.pkl")]
    else:
        vdir = "/u/yyu/stor/cnn/pureNoUnify/att/"
        vpath = os.path.join(vdir, "dict.pkl")

        train_files = [os.path.join(vdir, "train4_pass.pkl"),
                    os.path.join(vdir, "train4_qs.pkl"),
                    os.path.join(vdir, "train4_ans.pkl")]

        valid_files = [os.path.join(vdir, "valid4_pass.pkl"),
                    os.path.join(vdir, "valid4_qs.pkl"),
                    os.path.join(vdir, "valid4_ans.pkl")]
    return train_files, valid_files, vpath


def get_str_rep(lst, vocab):
    return [vocab[s] if s in vocab else vocab['UNK'] for s in lst]


def check_is_end(chks, i, vocab):
    test1 = "".join(get_str_rep(chks[i-1:i+3], vocab))
    test2 = "".join(get_str_rep(chks[i-3:i+1], vocab))
    rtn_val = True
    if test1 == "a.m." or test1 == 'p.m.':
        rtn_val = False
    elif test2 == 'p.m.' or test2 == 'a.m.':
        rtn_val = False
    return rtn_val


def parse_description_sents(seqs_x,
                            sent_end_toks,
                            eos_tok_id=None,
                            quote_tok_id=None,
                            vocab=None):
    batch = []
    words = []
    sents = []
    for i, desc in enumerate(seqs_x):
        words = []
        sents = []
        quote = False
        for j, wval in enumerate(desc):
            if j != len(desc) - 1 and desc[j+1] == quote_tok_id and quote:
                words.append(wval)
                quote = not quote
            elif wval in sent_end_toks and check_is_end(desc, j, vocab):
                words.append(wval)
                if len(words) > 1:
                    sents.append(words)
                words = []
                quote = False
            else:
                words.append(wval)
            if desc[j] == quote_tok_id:
                quote = not quote
        if len(sents) < 1:
            batch.append([desc])
        else:
            batch.append(sents)
    return batch


def load_data(train_file_paths=None,
              valid_file_paths=None,
              test_file_paths=None,
              batch_size=128,
              vocab=None,
              eyem=None,
              **sent_opts):

    train_data, valid_data, test_data = None, None, None

    if train_file_paths:
        print "Loading training files..."
        train_data = DataIterator(desc_file=train_file_paths[0],
                                  q_file=train_file_paths[1],
                                  ans_file=train_file_paths[2],
                                  batch_size=batch_size,
                                  shuffle=False,
                                  train_mode=True,
                                  vocab=vocab,
                                  eyem=eyem,
                                  **sent_opts)

    if valid_file_paths:
       print "Loading validation files..."
       valid_data = DataIterator(desc_file=valid_file_paths[0],
                                 q_file=valid_file_paths[1],
                                 ans_file=valid_file_paths[2],
                                 batch_size=batch_size,
                                 shuffle=False,
                                 permute_ents=False,
                                 use_infinite_loop=False,
                                 vocab=vocab,
                                 eyem=eyem,
                                 **sent_opts)

    if test_file_paths:
       print "Loading test files..."
       test_data = DataIterator(desc_file=test_file_paths[0],
                                q_file=test_file_paths[1],
                                ans_file=test_file_paths[2],
                                batch_size=batch_size,
                                permute_ents=False,
                                shuffle=False,
                                use_infinite_loop=False,
                                vocab=vocab,
                                eyem=eyem,
                                **sent_opts)
    return train_data, valid_data, test_data


logger = logging.getLogger(__name__)


class DataIterator(Iterator):

    def __init__(self,
                 ans_file,
                 desc_file,
                 q_file,
                 batch_size,
                 start=0,
                 stop=-1,
                 eyem=None,
                 shuffle=False,
                 use_sent_reps=False,
                 sent_end_tok_ids=None,
                 quote_tok_id=None,
                 eos_tok_id=None,
                 vocab=None,
                 permute_ents=True,
                 train_mode=False,
                 use_infinite_loop=False):

        if stop != -1:
            assert stop > start

        if permute_ents:
            assert vocab is not None

        self.vocab = vocab
        self.offset = 0
        self.start = start
        self.stop = stop
        self.batch_size = batch_size
        self.train_mode = train_mode
        self.use_sent_reps = use_sent_reps
        self.use_infinite_loop = use_infinite_loop
        self.sent_end_tok_ids = sent_end_tok_ids
        self.quote_tok_id = quote_tok_id
        self.eos_tok_id = eos_tok_id
        self.ans_file = ans_file
        self.desc_file = desc_file
        self.q_file = q_file
        self.data_len = 0
        self.shuffle = shuffle
        self.permute_ents = permute_ents
        self.done = False
        self.vsize = 0
        self.eyem=eyem
        self.__load_files()


    def __iter__(self):
        return self


    def next(self):
        ent_map = {}
        if self.permute_ents:
            ent_map = create_mappings(self.entities)

        map_ents = lambda sent: [ent_map[w] if w in ent_map else w for w in sent]
        map_batch_ents = lambda batch: [[ent_map[w] if w in ent_map else w \
                for w in sent] for sent in batch]

        flatten = lambda xxx: [x for xx in xxx for x in xx]
        retrieve_ents = lambda sent: [w for w in sent if w in self.entities]
        ent_mask = numpy.zeros((self.batch_size, self.vsize + 1)).astype("float32")
        if self.shuffle:

            inds = numpy.arange(self.start, self.stop)
            numpy.random.shuffle(inds)
            inds = inds[:self.batch_size]
            dvals, qvals, avals = [], [], []

            for i, ind in enumerate(inds):
                if len(dvals) > 0 or len(qvals) > 0:
                    dvals.append(map_ents(self.dvals[ind]))
                    qvals.append(map_ents(self.qvals[ind]))
                    avals.append(self.avals[ind][0])
                else:
                    dvals.append(map_ents(self.dvals[ind-1]))
                    qvals.append(map_ents(self.qvals[ind-1]))
                    avals.append(self.avals[ind-1][0])

                if i > 0 and len(dvals) <= 1:
                    import ipdb; ipdb.set_trace()
                if not self.train_mode:
                    desc_ents = retrieve_ents(dvals)
                    ent_mask[i] = create_mask_entities(desc_ents, self.eyem)

            avals = map_ents(avals)
            for i, dv in enumerate(dvals):
                if len(dv) < 1:
                    import ipdb; ipdb.set_trace()

            if self.use_sent_reps:
                dvals = parse_description_sents(dvals,
                                                self.sent_end_tok_ids,
                                                self.eos_tok_id,
                                                self.quote_tok_id,
                                                vocab=self.iadict)

            return dvals, qvals, avals, ent_mask
        else:
            if self.offset + self.batch_size > self.stop \
                    and not self.use_infinite_loop:
                self.done = True
                raise StopIteration
            elif self.use_infinite_loop and \
                    self.offset + self.batch_size > self.stop:
                first_part = slice(self.offset, self.stop)
                delta = self.batch_size - (self.stop - self.offset)
                second_part = slice(self.start, self.start + delta)
                dvals = map_batch_ents(self.dvals[first_part].extend(self.dvals[second_part]))
                avals = map_ents(self.avals[first_part].extend(self.avals[second_part]))
                qvals = map_batch_ents(self.qvals[first_part].extend(self.qvals[second_part]))
                self.offset = self.start + delta
                if self.use_sent_reps:
                    dvals = parse_description_sents(dvals,
                                                    self.sent_end_tok_ids,
                                                    self.eos_tok_id,
                                                    self.quote_tok_id,
                                                    vocab=self.iadict)
                if not self.train_mode:
                    for i, desc in enumerate(dvals):
                        desc_ents = retrieve_ents(dvals)
                        ent_mask[i] = create_mask_entities(desc_ents, self.eyem)

                return dvals, qvals, avals, ent_mask
            else:
                next_offset = self.offset + self.batch_size
                dvals = map_batch_ents(self.dvals[self.offset:next_offset])
                qvals = map_batch_ents(self.qvals[self.offset:next_offset])
                avals = map_ents(flatten(self.avals[self.offset:next_offset]))

                if self.use_sent_reps:
                    dvals = parse_description_sents(dvals,
                                                    self.sent_end_tok_ids,
                                                    self.eos_tok_id,
                                                    self.quote_tok_id,
                                                    vocab=self.iadict)

                if not self.train_mode:
                    for i, desc in enumerate(dvals):
                        desc_ents = retrieve_ents(dvals)
                        ent_mask[i] = create_mask_entities(desc_ents, self.eyem)

                self.offset = next_offset
                return dvals, qvals, avals, ent_mask


    def reset(self):
        self.offset = 0
        self.done = False


    def __load_files(self):
        logging.info("Started loading the files...")
        with open(self.ans_file, "r") as afile:
            self.avals = pkl.load(afile)

        with open(self.desc_file, "r") as dfile:
            self.dvals = pkl.load(dfile)

        with open(self.q_file, "r") as qfile:
            self.qvals = pkl.load(qfile)

        with open(self.vocab, "r") as vocabf:
            self.adict = pkl.load(vocabf)

        self.iadict = {v:k for k, v in self.adict.iteritems()}
        self.data_len = len(self.avals)

        if self.stop == -1:
            self.stop = self.data_len

        self.vsize = len(self.adict)
        self.entities = get_entities(self.adict)
        logging.info("Loaded the files...")
