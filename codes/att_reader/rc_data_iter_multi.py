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

    if maxlen != None:
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

def load_data(path=None, valid_path=None, test_path=None, batch_size=128):
    '''
       Loads the dataset.
    '''
    if path is None:
        path='/u/yyu/stor/caglar/rc-data/cnn/cnn_train_data.h5'

    print "Using training data ", path
    #############
    # LOAD DATA #
    #############

    print '... initializing data iterators'
    train = PytablesRCDataIterator(batch_size, path, use_infinite_loop=False)
    valid = PytablesRCDataIterator(batch_size, valid_path, shuffle=False,
                                   use_infinite_loop=False) if valid_path else None
    test = PytablesRCDataIterator(batch_size, test_path, shuffle=False,
                                  use_infinite_loop=False) if test_path else None
    return train, valid, test


def get_length(path):
    if tables.__version__[0] == '2':
        target_table = tables.openFile(path, 'r')
        target_index = target_table.getNode('/indices')
    else:
        target_table = tables.open_file(path, 'r')
        target_index = target_table.get_node('/indices')

    return target_index.shape[0]


lock = threading.RLock()
def synchronized_open_file(*args, **kwargs):
    if tables.__version__[0] == '2':
        tbf = tables.openFile(*args, **kwargs)
    else:
        tbf = tables.open_file(*args, **kwargs)
    return tbf


class PytablesRCDataFetcher(threading.Thread):
    def __init__(self, parent, start_offset, max_offset=-1):
        threading.Thread.__init__(self)
        self.parent = parent
        self.start_offset = start_offset
        self.max_offset = max_offset
        self.offset = 0
        self._stop = threading.Event()


    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()

    def run(self):

        diter = self.parent

        driver = None
        if diter.can_fit:
            driver = "H5FD_CORE"

        max_qlen = 0
        max_dlen = 0
        try:
            target_table = synchronized_open_file(diter.target_file, 'r',
                                                    driver=driver)

            if tables.__version__[0] == '2':
                d_data, d_index = (target_table.getNode(diter.dtable_name),
                    target_table.getNode(diter.dindex_name))
                q_data, q_index = (target_table.getNode(diter.qtable_name),
                    target_table.getNode(diter.qindex_name))
                a_data, a_index = (target_table.getNode(diter.atable_name),
                    target_table.getNode(diter.aindex_name))
            else:
                d_data, d_index = (target_table.get_node(diter.dtable_name),
                    target_table.get_node(diter.dindex_name))
                q_data, q_index = (target_table.get_node(diter.qtable_name),
                    target_table.get_node(diter.qindex_name))
                a_data, a_index = (target_table.get_node(diter.atable_name),
                    target_table.get_node(diter.aindex_name))

            data_len = d_index.shape[0]
            self.offset = self.start_offset

            if self.offset == -1:
                self.offset = 0
                self.start_offset = self.offset
                if diter.shuffle:
                    self.offset = numpy.random.randint(low=self.start_offset, hight=self.stop_offset)

            logger.debug("{} entries".format(data_len))
            logger.debug("Starting from the entry {}".format(self.offset))

            while not diter.exit_flag:
                last_batch = False
                desc_ngrams = []
                q_ngrams = []
                ans = []

                while len(desc_ngrams) < diter.batch_size:
                    if self.offset == data_len or self.offset == self.max_offset:
                        if diter.use_infinite_loop:
                            self.offset = self.start_offset
                        else:
                            last_batch = True
                            break

                    dlen, dpos = d_index[self.offset]['length'], d_index[self.offset]['pos']
                    qlen, qpos = q_index[self.offset]['length'], q_index[self.offset]['pos']
                    apos = a_index[self.offset]['pos']

                    if dlen > max_dlen:
                        max_dlen = dlen

                    if qlen > max_qlen:
                        max_qlen = qlen

                    self.offset += 1

                    desc_ngrams.append(d_data[dpos:dpos+dlen])
                    q_ngrams.append(q_data[qpos:qpos+qlen])
                    ans.append(a_data[apos])
        except Exception as e:
            diter.queue.put(e)
        else:
            if len(desc_ngrams):
                diter.queue.put([desc_ngrams, q_ngrams, ans, max_dlen, max_qlen])

            if last_batch:
                diter.queue.put([None, None, None])
                diter.queue.task_done()
                return
        finally:
            print "Done"

    def reset(self, start_offset=None, max_offset=None):
        if start_offset is not None:
            self.start_offset = start_offset
        else:
            self.start_offset = 0

        if max_offset is not None:
            self.max_offset = max_offset
        self.offset = self.start_offset


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
                 queue_size=1000,
                 cache_size=1000,
                 shuffle=True,
                 use_infinite_loop=True):

        args = locals()
        args.pop("self")
        self.__dict__.update(args)
        self.queue = None
        self.gather = None
        self.started = False
        self.exit_flag = False

    def start_it(self, start_offset=0, max_offset=-1):
        self.started = True
        if start_offset == 0 and self.start > 0:
            start_offset = self.start

        if max_offset == -1 and self.stop > start_offset:
            max_offset = self.stop

        self.queue = Queue.Queue(maxsize=self.queue_size)
        self.gather = PytablesRCDataFetcher(self, start_offset, max_offset)
        self.gather.daemon = True
        self.gather.start()

        """
        if self.queue is None or self.gather is None:
            self.gather = PytablesRCDataFetcher(self, start_offset, max_offset)
            self.gather.daemon = True
            self.gather.start()
        else:
            if self.gather.stopped():
                self.gather = PytablesRCDataFetcher(self, start_offset, max_offset)
                self.gather.daemon = True
                self.gather.start()
            else:
                self.gather.stop()
                self.gather = PytablesRCDataFetcher(self, start_offset, max_offset)
                self.gather.daemon = True
                self.gather.start()
        """

    def __del__(self):
        if hasattr(self, 'gather'):
            self.gather.exitFlag = True
            self.gather.join()

    def __iter__(self):
        return self

    def next(self):
        if not hasattr(self, "queue"):
            raise ValueError("You need to start the iterator first via start()")
        else:
            batch = self.queue.get()

        if not batch or batch[0] is None:
            raise StopIteration
        desc = batch[0]
        if isinstance(desc, str):
            print >> sys.stderr, desc
            import ipdb; ipdb.set_trace()
        else:
            q = batch[1]
            ans = batch[2]
            max_dlen = batch[3] + 1
            max_qlen = batch[4] + 1
            return desc, q, ans, max_dlen, max_qlen

    def reset(self):
        assert self.started, "You should start the iterator first!"
        self.gather.reset()

if __name__ == "__main__":
    train, _, _ = load_data()
    train.start()
    x, y, z, d, q = next(train)
