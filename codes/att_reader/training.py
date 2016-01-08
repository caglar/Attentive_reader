import theano
from theano import tensor
import numpy
from utils import itemlist
from core.utils import sharedX
from collections import OrderedDict

profile = False


# optimizers
def adam(lr, tparams, grads, inp, cost, errors):
    gshared = OrderedDict({p: sharedX(p.get_value() * 0.,
                           name='%s_grad' % p.name)
                           for p, g in grads.iteritems()})

    gsup = [(gshared[p], g) for p, g in grads.iteritems()]
    gnorm = get_norms(grads.values())
    pnorm = get_norms(tparams.values())
    f_grad_shared = theano.function(inp,
                                    [cost, errors,
                                        gnorm, pnorm],
                                    updates=gsup,
                                    profile=profile)

    lr0 = lr
    b1 = 0.1
    b2 = 0.001
    e = 1e-8

    updates = []
    i = sharedX(numpy.float32(0.))
    i_t = i + 1.

    fix1 = 1.0 - (1 - b1)**(i_t)
    fix2 = 1.0 - (1 - b2)**(i_t)

    lr_t = lr0 * (tensor.sqrt(fix2) / fix1)
    up_list = []

    for p in tparams.values():
        g = gshared[p]
        m = sharedX(p.get_value() * 0.)
        v = sharedX(p.get_value() * 0.)
        m_t = (b1 * g) + ((1. - b1) * m)
        v_t = (b2 * tensor.sqr(g)) + ((1. - b2) * v)
        g_t = m_t / (tensor.sqrt(v_t) + e)
        up_list.append(lr_t * g_t)
        p_t = p - (lr_t * g_t)
        updates.append((m, m_t))
        updates.append((v, v_t))
        updates.append((p, p_t))

    updates.append((i, i_t))
    upnorm = get_norms(up_list)
    f_update = theano.function([lr],
                               [upnorm],
                               updates=updates,
                               on_unused_input='ignore',
                               profile=profile)

    return f_grad_shared, f_update


def adadelta(lr, tparams, grads, inp, cost, errors):
    gnorm = get_norms(grads)
    pnorm = get_norms(tparams.values())

    zipped_grads = [sharedX(p.get_value() * numpy.float32(0.),
                                  name='%s_grad'%k)
                    for k, p in tparams.iteritems()]

    running_up2 = [sharedX(p.get_value() * numpy.float32(0.),
                                 name='%s_rup2'%k)
                   for k, p in tparams.iteritems()]

    running_grads2 = [sharedX(p.get_value() * numpy.float32(0.),
                                    name='%s_rgrad2'%k)
                      for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in \
            zip(zipped_grads, grads)]

    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) \
                for rg2, g in \
                zip(running_grads2, grads)]

    f_grad_shared = theano.function(inp,
                                    [cost, errors, gnorm, pnorm], \
                                    updates=zgup + rg2up)

    updir = [-tensor.sqrt(ru2 + 1e-6) / \
                tensor.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 \
                in zip(zipped_grads, running_up2, running_grads2)]

    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) \
               for ru2, ud in zip(running_up2, updir)]

    param_up = [(p, p + ud) for p, ud in \
                   zip(itemlist(tparams), updir)]

    upnorm = get_norms(updir)
    f_update = theano.function([lr], [upnorm],
                               updates=ru2up+param_up,
                               on_unused_input='ignore',
                               profile=profile)

    return f_grad_shared, f_update


def get_norms(gs):
    if isinstance(gs, list):
        gnorm = tensor.sqrt(sum([(g[1]**2).sum()
                               if isinstance(g, list) and \
                                       len(g) > 1 else (g**2).sum()
                               for g in gs]))
    elif isinstance(gs, dict) or isinstance(gs, OrderedDict):
        gnorm = tensor.sqrt(sum([(g**2).sum() for p, g in gs.iteritems()]))

    return gnorm


def rmsprop(lr, tparams, grads, inp, cost, errors):
    zipped_grads = [sharedX(p.get_value() * numpy.float32(0.), \
            name='%s_grad'%k) for k, p in tparams.iteritems()]

    running_grads = [sharedX(p.get_value() * numpy.float32(0.), \
            name='%s_rgrad'%k) for k, p in tparams.iteritems()]

    running_grads2 = [sharedX(p.get_value() * numpy.float32(0.), \
            name='%s_rgrad2'%k) for k, p in tparams.iteritems()]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, grads)]
    rgup = [(rg, 0.95 * rg + 0.05 * g) for rg, g in zip(running_grads, grads)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g \
            in zip(running_grads2, grads)]

    pnorm = get_norms(tparams.values())
    gnorm = get_norms(grads)

    f_grad_shared = theano.function(inp,
                                    [cost, errors, gnorm, pnorm],
                                    updates=zgup+rgup+rg2up,
                                    profile=profile)

    updir = [sharedX(p.get_value() * numpy.float32(0.),
                     name='%s_updir'%k) \
                     for k, p in tparams.iteritems()]

    updir_new = [(ud, 0.9 * ud - lr * zg / \
                 tensor.maximum(tensor.sqrt(rg2 - rg ** 2 + 1e-8)), 1e-8) \
                 for ud, zg, rg, rg2 in zip(updir, zipped_grads, \
                        running_grads, running_grads2)]

    param_up = [(p, p + udn[1]) for p, udn in \
                zip(itemlist(tparams), updir_new)]

    upnorm = get_norms(updir_new)
    f_update = theano.function([lr],
                               [upnorm],
                               updates=updir_new+param_up,
                               on_unused_input='ignore',
                               profile=profile)

    return f_grad_shared, f_update


def sgd(lr, tparams, grads, x, mask, y, cost, errors):
    gshared = [sharedX(p.get_value() * 0.,
                       name='%s_grad'%k) for k, p \
                               in tparams.iteritems()]
    gsup = [(gs, g) for gs, g in zip(gshared, grads)]

    pnorm = get_norms(tparams.values())
    gnorm = get_norms(grads)

    f_grad_shared = theano.function([x, mask, y],
                                    [cost, errors, gnorm, pnorm],
                                    updates=gsup, profile=profile)

    pup = [(p, p - lr * g) for p, g in zip(itemlist(tparams), gshared)]
    upnorm = lr*gnorm
    f_update = theano.function([lr], [upnorm], updates=pup, profile=profile)

    return f_grad_shared, f_update
