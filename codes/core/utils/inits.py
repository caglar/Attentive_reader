import os

import random
import string
import copy as pycopy
import re

#Groundhog related imports
import numpy
np = numpy

import theano
import theano.tensor as TT
import inspect
from ..commons import EPS, global_rng


def sample_zeros(sizeX, sizeY, sparsity, scale, rng):
    return numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)


def sample_weights(sizeX, sizeY, sparsity, scale, rng):
    """
    Initialization that fixes the largest singular value.
    """
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.uniform(low=-scale, high=scale, size=(sparsity,))
        vals_norm = numpy.sqrt((new_vals**2).sum())
        new_vals = scale*new_vals/vals_norm
        values[dx, perm[:sparsity]] = new_vals
    _, v, _ = numpy.linalg.svd(values)
    values = scale * values/v[0]
    return values.astype(theano.config.floatX)


def sample_weights_uni_xav(sizeX, sizeY, rng):
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    fantot = sizeX + sizeY
    scale = np.sqrt(6 / float(fantot))
    values = rng.uniform(low=-scale, high=scale,
                         size=(sizeX, sizeY))
    return values.astype(theano.config.floatX)


def sample_weights_uni_rect(sizeX, sizeY, rng):
    sizeX = int(sizeX)
    sizeY = int(sizeY)
    fantot = sizeX + sizeY
    scale = np.sqrt(2 / float(fantot))
    values = rng.uniform(low=-scale, high=scale,
                         size=(sizeX, sizeY))
    return values.astype(theano.config.floatX)


def sample_weights_classic(sizeX, sizeY, sparsity, scale, rng=None):
    if rng is None:
        rng = global_rng

    sizeX = int(sizeX)
    sizeY = int(sizeY)

    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimum(sizeY, sparsity)

    sparsity = numpy.minimum(sizeY, sparsity)
    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    for dx in xrange(sizeX):
        perm = rng.permutation(sizeY)
        new_vals = rng.normal(loc=0, scale=scale, size=(sparsity,))
        values[dx, perm[:sparsity]] = new_vals
    return values.astype(theano.config.floatX)


def sample_weights_orth(sizeX, sparsity, scale=1.0, rng=None):
    sizeX = int(sizeX)
    sizeY = sizeX

    assert sizeX == sizeY, 'for orthogonal init, sizeX == sizeY'

    if sparsity < 0:
        sparsity = sizeY
    else:
        sparsity = numpy.minimum(sizeY, sparsity)

    values = numpy.zeros((sizeX, sizeY), dtype=theano.config.floatX)
    values = rng.normal(loc=0, scale=scale, size=(sizeX, sizeY))
    u,s,v = numpy.linalg.svd(values)
    values = u.dot(v.T)
    #values = u * scale
    return values.astype(theano.config.floatX)


def sample_3dtensor_orth(sizeX, bs, sparsity, scale=1.0, rng=None):
    orth3DTens = np.random.uniform(-0.1, 0.1,
                                   (bs, sizeX, sizeX)).astype("float32")
    for i in xrange(bs):
        u, s, v = np.linalg.svd(orth3DTens[i])
        orth3DTens[i] = u.dot(v.T)
    return orth3DTens


def init_bias(size, scale, rng):
    return numpy.ones((size,), dtype=theano.config.floatX)*scale


