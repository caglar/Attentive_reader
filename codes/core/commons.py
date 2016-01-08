import logging
import sys

import numpy

import theano
import theano.tensor as TT
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

EPS = 1e-6
DEFAULT_SEED = 3
floatX = theano.config.floatX

global_rng = numpy.random.RandomState(DEFAULT_SEED)
global_trng = RandomStreams(DEFAULT_SEED)

Sigmoid = lambda x, use_noise=0: TT.nnet.sigmoid(x)
Softmax = lambda x : TT.nnet.softmax(x)
Tanh = lambda x, use_noise=0: TT.tanh(x)

#Rectifier nonlinearities
Rect = lambda x, use_noise=0: 0.5 * (x + abs(x))
Leaky_Rect = lambda x, leak=0.95, use_noise=0: ((1 + leak) * x + (1 - leak) * abs(x)) * 0.5
Trect = lambda x, use_noise=0: Rect(Tanh(x + EPS))
Trect_dg = lambda x, d, use_noise=0: Rect(Tanh(d*x))
Softmax = lambda x: TT.nnet.softmax(x)
Linear = lambda x: x

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def ELU(x, alpha=1.0):
    switch = x >= 0.
    return switch*x + (1. - switch) * alpha * (TT.exp(x) - 1.)


#Change this to change the location of parent directory where your models will be dumped into.
SAVE_DUMP_FOLDER="./"
