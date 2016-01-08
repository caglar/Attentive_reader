import theano
import theano.tensor as tensor
import numpy

from att_reader.utils import prfx, norm_weight, ortho_weight
from core.utils import dot, sharedX
from core.commons import Sigmoid, Tanh, Rect, global_trng, Linear, ELU

"""
    We have functions to create the layers and initialize them.
"""


profile = False
layers = {
          'ff': ('param_init_fflayer', 'fflayer'),
          'gru': ('param_init_gru', 'gru_layer'),
          'gru_cond': ('param_init_gru_cond', 'gru_cond_layer'),
          'lstm': ('param_init_lstm', 'lstm_layer'),
          'lstm_tied': ('param_init_lstm_tied', 'lstm_tied_layer'),
          }


# layer
def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(options,
                       params,
                       prefix='ff',
                       nin=None,
                       nout=None,
                       ortho=True,
                       use_bias=True):
    if nin is None:
        nin = options['dim_proj']

    if nout is None:
        nout = options['dim_proj']

    params[prfx(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)

    if use_bias:
        params[prfx(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    return params


def fflayer(tparams,
            state_below,
            options,
            prefix='rconv',
            use_bias=True,
            activ='lambda x: tensor.tanh(x)',
            **kwargs):

    if use_bias:
        return eval(activ)(dot(state_below, tparams[prfx(prefix, 'W')]) + tparams[prfx(prefix, 'b')])
    else:
        return eval(activ)(dot(state_below, tparams[prfx(prefix, 'W')]))


# GRU layer
def param_init_gru(options,
                   params,
                   prefix='gru',
                   nin=None,
                   dim=None,
                   hiero=False):

    if nin is None:
        nin = options['dim_proj']

    if dim is None:
        dim = options['dim_proj']

    if not hiero:
        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[prfx(prefix, 'W')] = W
        params[prfx(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')

    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[prfx(prefix, 'U')] = U
    Wx = norm_weight(nin, dim)
    params[prfx(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[prfx(prefix, 'Ux')] = Ux
    params[prfx(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')
    return params


def gru_layer(tparams,
              state_below,
              options,
              prefix='gru',
              mask=None,
              nsteps=None,
              truncate=None,
              init_state=None,
              **kwargs):

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[prfx(prefix, name)]
    dim = param('Ux').shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    if mask.ndim == 3 and mask.ndim == state_below.ndim:
        mask = mask.reshape((mask.shape[0], \
                mask.shape[1] * mask.shape[2])).dimshuffle(0, 1, 'x')
    elif mask.ndim == 2:
        mask = mask.dimshuffle(0, 1, 'x')

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = dot(state_below, param('W')) + param('b')
    state_belowx = dot(state_below, param('Wx')) + param('bx')

    # initial/previous state
    if init_state is None:
        if not options['learn_h0']:
            init_state = tensor.alloc(0., n_samples, dim)
        else:
            init_state0 = sharedX(numpy.zeros((options['dim'])),
                                 name=prfx(prefix, "h0"))
            init_state = tensor.concatenate([[init_state0] \
                                                for i in xrange(options['batch_size'])],
                                            axis=0)
            tparams[prfx(prefix, 'h0')] = init_state0

    U = tparams[prfx(prefix, 'U')]
    Ux = tparams[prfx(prefix, 'Ux')]

    def _step_slice(mask, sbelow, sbelowx, sbefore, U, Ux):
        preact = dot(sbefore, U)
        preact += sbelow

        r = Sigmoid(_slice(preact, 0, dim))
        u = Sigmoid(_slice(preact, 1, dim))

        preactx = dot(r * sbefore, Ux)

        # preactx = preactx
        preactx = preactx + sbelowx

        h = Tanh(preactx)

        h = u * sbefore + (1. - u) * h
        h = mask[:, None] * h + (1. - mask)[:, None] * sbefore

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[init_state],
                                non_sequences=[U, Ux],
                                name=prfx(prefix, '_layers'),
                                n_steps=nsteps,
                                truncate_gradient=truncate,
                                profile=profile,
                                strict=True)
    rval = [rval]
    return rval


# LSTM layer
def param_init_lstm(options,
                    params,
                    prefix='lstm',
                    nin=None,
                    dim=None):

    if nin is None:
        nin = options['dim_proj']

    if dim is None:
        dim = options['dim_proj']

    W = numpy.concatenate([norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim),
                           norm_weight(nin,dim)],
                           axis=1)

    params[prfx(prefix,'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)],
                           axis=1)

    params[prfx(prefix,'U')] = U
    params[prfx(prefix,'b')] = numpy.zeros((4 * dim,)).astype('float32')

    return params


def lstm_layer(tparams, state_below,
               options,
               prefix='lstm',
               mask=None, one_step=False,
               init_state=None,
               init_memory=None,
               nsteps=None,
               **kwargs):

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[prfx(prefix, name)]
    dim = param('U').shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        if not options['learn_h0']:
            init_state = tensor.alloc(0., n_samples, dim)
        else:
            init_state0 = sharedX(numpy.zeros((options['dim'])),
                                 name=prfx(prefix, "h0"))
            init_state = tensor.alloc(init_state0, n_samples, dim)
            tparams[prfx(prefix, 'h0')] = init_state0

    U = param('U')
    b = param('b')
    W = param('W')
    non_seqs = [U, b, W]

    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(mask, sbelow, sbefore, cell_before, *args):
        preact = dot(sbefore, param('U'))
        preact += sbelow
        preact += param('b')

        i = Sigmoid(_slice(preact, 0, dim))
        f = Sigmoid(_slice(preact, 1, dim))
        o = Sigmoid(_slice(preact, 2, dim))
        c = Tanh(_slice(preact, 3, dim))

        c = f * cell_before + i * c
        c = mask * c + (1. - mask) * cell_before
        h = o * tensor.tanh(c)
        h = mask * h + (1. - mask) * sbefore

        return h, c

    lstm_state_below = dot(state_below, param('W')) + param('b')
    if state_below.ndim == 3:
        lstm_state_below = lstm_state_below.reshape((state_below.shape[0],
                                                     state_below.shape[1],
                                                     -1))
    if one_step:
        mask = mask.dimshuffle(0, 'x')
        h, c = _step(mask, lstm_state_below, init_state, init_memory)
        rval = [h, c]
    else:
        if mask.ndim == 3 and mask.ndim == state_below.ndim:
            mask = mask.reshape((mask.shape[0], \
                                 mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
        elif mask.ndim == 2:
            mask = mask.dimshuffle(0, 1, 'x')

        rval, updates = theano.scan(_step,
                                    sequences=[mask, lstm_state_below],
                                    outputs_info = [init_state,
                                                    init_memory],
                                    name=prfx(prefix, '_layers'),
                                    non_sequences=non_seqs,
                                    strict=True,
                                    n_steps=nsteps)
    return rval


# LSTM layer
def param_init_lstm_tied(options,
                         params,
                         prefix='lstm_tied',
                         nin=None,
                         dim=None):

    if nin is None:
        nin = options['dim_proj']

    if dim is None:
        dim = options['dim_proj']

    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)

    params[prfx(prefix, 'W')] = W
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim),
                           ortho_weight(dim)], axis=1)

    params[prfx(prefix, 'U')] = U
    params[prfx(prefix, 'b')] = numpy.zeros((3 * dim,)).astype('float32')

    return params


def lstm_tied_layer(tparams,
                    state_below,
                    options,
                    prefix='lstm_tied',
                    mask=None,
                    one_step=False,
                    init_state=None,
                    init_memory=None,
                    nsteps=None,
                    **kwargs):

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    param = lambda name: tparams[prfx(prefix, name)]
    dim = param('U').shape[0]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    # initial/previous state
    if init_state is None:
        if not options['learn_h0']:
            init_state = tensor.alloc(0., n_samples, dim)
        else:
            init_state0 = sharedX(numpy.zeros((options['dim'])),
                                 name=prfx(prefix, "h0"))
            init_state = tensor.concatenate([[init_state0] \
                                                for i in xrange(options['batch_size'])],
                                            axis=0)
            tparams[prfx(prefix, 'h0')] = init_state0

    # initial/previous memory
    if init_memory is None:
        init_memory = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    def _step(mask, sbelow, sbefore, cell_before):
        preact = dot(sbefore, param('U'))
        preact += sbelow
        preact += tparams[prfx(prefix, 'b')]

        f = Sigmoid(_slice(preact, 0, dim))
        o = Sigmoid(_slice(preact, 1, dim))
        c = Tanh(_slice(preact, 2, dim))

        c = f * cell_before + (1 - f) * c
        c = mask * c + (1. - mask) * cell_before
        h = o * tensor.tanh(c)
        h = mask * h + (1. - mask) * sbefore

        return h, c

    state_below = dot(state_below, param('W')) + param('b')

    if one_step:
        mask = mask.dimshuffle(0, 'x')
        h, c = _step(mask, state_below, init_state, init_memory)
        rval = [h, c]
    else:
        if mask.ndim == 3 and mask.ndim == state_below.ndim:
            mask = mask.reshape((mask.shape[0], mask.shape[1]*mask.shape[2])).dimshuffle(0, 1, 'x')
        elif mask.ndim == 2:
            mask = mask.dimshuffle(0, 1, 'x')
        rval, updates = theano.scan(_step,
                                    sequences=[mask, state_below],
                                    outputs_info=[init_state,
                                                  init_memory],
                                    name=prfx(prefix, '_layers'),
                                    n_steps=nsteps)
    return rval


# Conditional GRU layer with Attention
def param_init_gru_cond(options,
                        params,
                        prefix='gru_cond',
                        nin=None,
                        dim=None,
                        dimctx=None):

    if nin is None:
        nin = options['dim']
    if dim is None:
        dim = options['dim']
    if dimctx is None:
        dimctx = options['dim']

    params = param_init_gru(options,
                            params,
                            prefix,
                            nin=nin,
                            dim=dim)
    # context to LSTM
    Wc = norm_weight(dimctx, dim*2)
    params[prfx(prefix, 'Wc')] = Wc

    Wcx = norm_weight(dimctx, dim)
    params[prfx(prefix, 'Wcx')] = Wcx

    # attention: prev -> hidden
    Wi_att = norm_weight(nin, dimctx)
    params[prfx(prefix, 'Wi_att')] = Wi_att

    # attention: context -> hidden
    Wc_att = norm_weight(dimctx)
    params[prfx(prefix, 'Wc_att')] = Wc_att

    # attention: LSTM -> hidden
    Wd_att = norm_weight(dim, dimctx)
    params[prfx(prefix, 'Wd_att')] = Wd_att

    # attention: hidden bias
    b_att = numpy.zeros((dimctx,)).astype('float32')
    params[prfx(prefix, 'b_att')] = b_att

    # attention:
    U_att = norm_weight(dimctx, 1)
    params[prfx(prefix, 'U_att')] = U_att
    c_att = numpy.zeros((1,)).astype('float32')
    params[prfx(prefix, 'c_tt')] = c_att

    return params


def gru_cond_layer(tparams,
                   state_below,
                   options,
                   prefix='gru',
                   mask=None,
                   context=None,
                   one_step=False,
                   init_memory=None,
                   init_state=None,
                   context_mask=None,
                   nsteps=None,
                   **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    if nsteps is None:
        nsteps = state_below.shape[0]

    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[prfx(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, 'Context must be 3-d: #annotation x #sample x dim'
    pctx_ = dot(context, tparams[prfx(prefix, 'Wc_att')]) + tparams[prfx(prefix, 'b_att')]

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = dot(state_below, tparams[prfx(prefix, 'Wx')]) + \
            tparams[prfx(prefix, 'bx')]

    state_below_ = dot(state_below, tparams[prfx(prefix, 'W')]) + \
            tparams[prfx(prefix, 'b')]

    state_belowc = dot(state_below, tparams[prfx(prefix, 'Wi_att')])

    def _step_slice(mask,
                    sbelow,
                    sbelowx,
                    xc_, sbefore,
                    ctx_, alpha_,
                    pctx_, cc_,
                    U, Wc,
                    Wd_att, U_att,
                    c_tt, Ux, Wcx):
        # attention
        pstate_ = dot(sbefore, Wd_att)
        pctx__ = pctx_ + pstate_[None, :, :]
        pctx__ += xc_
        pctx__ = Tanh(pctx__)
        alpha = dot(pctx__, U_att)+c_tt
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask

        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)
        # current context

        preact = dot(sbefore, U)
        preact += sbelow
        preact += dot(ctx_, Wc)
        preact = Sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = dot(sbefore, Ux)
        preactx *= r
        preactx += sbelowx
        preactx += dot(ctx_, Wcx)

        h = Tanh(preactx)

        h = u * sbefore + (1. - u) * h
        h = mask[:, None] * h + (1. - mask)[:, None] * sbefore

        return h, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[prfx(prefix, 'U')],
                   tparams[prfx(prefix, 'Wc')],
                   tparams[prfx(prefix, 'Wd_att')],
                   tparams[prfx(prefix, 'U_att')],
                   tparams[prfx(prefix, 'c_tt')],
                   tparams[prfx(prefix, 'Ux')],
                   tparams[prfx(prefix, 'Wcx')]]

    if one_step:
        rval = _step(*(seqs+[init_state, None, None, pctx_, context]+shared_vars))
    else:
        rval, updates = theano.scan(_step,
                                    sequences=seqs,
                                    outputs_info=[init_state,
                                                  tensor.alloc(0., n_samples, context.shape[2]),
                                                  tensor.alloc(0., n_samples, context.shape[0])],
                                    non_sequences=[pctx_,
                                                   context]+shared_vars,
                                    name=prfx(prefix, '_layers'),
                                    n_steps=nsteps,
                                    profile=profile,
                                    strict=True)
    return rval


def dropout_layer(state_before,
                  use_noise,
                  p=0.5):
    proj = tensor.switch(use_noise,
            state_before * global_trng.binomial(state_before.shape,
                                                p=p, n=1,
                                                dtype=state_before.dtype),
                                                state_before * p)
    return proj
