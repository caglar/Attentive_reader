import warnings

import numpy

import theano
from theano import tensor

from collections import OrderedDict
from core.commons import global_rng


def create_mask_entities(ents, I):
    mask = I[ents].sum(0).astype("float32")
    return mask


def create_entity_mask(vocab, I):
   ents = numpy.array(get_entities(vocab))
   return create_mask_entities(ents, I)


def negentropy(ps):
    return sum([(p*numpy.log(p + 1e-8)).sum((0, -1)).mean() \
            for p in ps]) / float(len(ps))


def reset_train_vals():
    return 0.0, 0.0, 0.0


# Create the model name for the
# models based on the options.
def create_model_name(prefix, options):
    name = list(prefix)[:-4]
    ext = list(prefix)[-4:]
    used = 0
    for i, (x, y) in enumerate(options.iteritems()):
        name_ = str(x)
        val_ = str(y)
        if name_ != "saveto" and name_ != "model" \
                and name_ != "model_dir" and \
                len(name_) <= 12 and len(val_) <= 7:
            used += 1

            name_ = name_.replace("_", "")
            if i == len(options) - 1:
                name += ["%s_%s" % (name_[:8], val_)]
            else:
                name += ["%s_%s_" % (name_[:8], val_)]

        used += len(name[-1])

        # Ensure the file name limit
        # is not exceeding the
        # filename length limit in OS:
        if used >= (240 - len(ext)):
            break
    return "".join(name + ext)


def get_entities(vocab):
    return [v for k, v in vocab.iteritems() \
            if "@entity" in k]


def create_mappings(entities):
    ent_len = len(entities)
    inds = numpy.arange(ent_len)
    global_rng.shuffle(inds)
    ent_map = {}

    for i, entity in enumerate(entities):
        ent_map[entity] = entities[inds[i]]
    return ent_map


# push parameters to Theano
# shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)


# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params


# get the list of parameters:_
# Note that tparams must be OrderedDict
def itemlist(tparams):
    return [vv for kk, vv in tparams.iteritems()]


# dropout
# make prefix-appended name
def prfx(pp, name):
    return '%s_%s' % (pp, name)


# initialize Theano shared variables
# according to the initial parameters
def init_tparams(params):
    tparams = OrderedDict()
    for kk, pp in params.iteritems():
        tparams[kk] = theano.shared(params[kk], name=kk)
    return tparams


# load parameters
def load_params(path, params):
    pp = numpy.load(path)
    for kk, vv in params.iteritems():
        if kk not in pp:
            warnings.warn('%s is not in the archive' % kk)
            continue
        params[kk] = pp[kk]
    return params


# some utilities
def ortho_weight(ndim, scale=0.05):
    numpy.random.seed(123)
    fanin = 2 * ndim + 1
    fanout = fanin
    minv = -numpy.sqrt(6.0 / float(fanin))
    maxv = numpy.sqrt(6.0 / float(fanout))
    W = numpy.random.uniform(minv, maxv, (ndim, ndim))
    u, s, v = numpy.linalg.svd(W)
    return (scale * u.dot(v.T)).astype('float32')


def norm_weight(nin,
                nout=None,
                scale=0.01,
                ortho=True):
    numpy.random.seed(123)
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin)
    else:
        fanin = nin + nout + 1
        fanout = fanin
        minv = -numpy.sqrt(6.0 / float(fanin))
        maxv = numpy.sqrt(6.0 / float(fanout))
        W = numpy.random.uniform(minv, maxv, (nin, nout))
    return W.astype('float32')


def norm_vec(dim):
    numpy.random.seed(123)
    minv = -numpy.sqrt(1.0/float(dim))
    maxv = -minv
    return numpy.random.uniform(minv,
                                maxv,
                                (dim,)).astype("float32")


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def Masked_Softmax(x, mask=None, ax=1):
    e_x = mask * tensor.exp(x - x.max(axis=ax, keepdims=True))
    sm = e_x / e_x.sum(axis=ax, keepdims=True)
    return sm
