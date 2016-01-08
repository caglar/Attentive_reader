import logging

from collections import OrderedDict
from att_reader.utils import norm_weight, ortho_weight, \
                            norm_vec, Masked_Softmax
from att_reader.layers import get_layer, dropout_layer

import numpy

import theano
from theano import tensor
from theano.tensor import concatenate
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import sys

from core.utils import dot
from core.commons import Sigmoid, Tanh, Softmax, ELU
from core.costs import nll_simple, multiclass_hinge_loss

from collections import OrderedDict


profile = False

logger = logging.getLogger(__name__)

# initialize all parameters
def init_params(options):
    params = OrderedDict()

    # embedding
    params['Wemb_word'] = norm_weight(options['n_words_q'],
                                      options['dim_word_desc'])

    mult = 2
    if options['ms_nlayers'] > 1 and (options['encoder_desc'] == 'lstm_ms' or \
            options['encoder_desc'] == 'lstm_max_ms'):

        mult = options['ms_nlayers']
        if options['use_bidir']:
            mult *= 2

    if options['use_dq_sims']:
        params['ff_att_bi_dq'] = \
                norm_weight(mult * options['dim'],
                            mult * options['dim'])

    params['ff_att_proj'] = norm_vec(options['dim'])

    # encoder: bidirectional RNN
    params = get_layer(options['encoder_desc_word'])[0](options,
                                                        params,
                                                        prefix='encoder_desc_word',
                                                        nin=options['dim_word_desc'],
                                                        dim=options['dim'])
    params = get_layer(options['encoder_q'])[0](options,
                                                params,
                                                prefix='encoder_q',
                                                nin=options['dim_word_q'],
                                                dim=options['dim'])

    if options['use_bidir']:
        params = get_layer(options['encoder_desc_word'])[0](options,
                                                            params,
                                                            prefix='encoder_desc_word_r',
                                                            nin=options['dim_word_desc'],
                                                            dim=options['dim'])
        params = get_layer(options['encoder_q'])[0](options,
                                                    params,
                                                    prefix='encoder_q_r',
                                                    nin=options['dim_word_q'],
                                                    dim=options['dim'])


    if options['use_sent_reps']:
        params['Wemb_sent'] = norm_weight(mult * options['dim'],
                                          mult * options['dim'])
        # encoder: bidirectional RNN
        params = get_layer(options['encoder_desc_sent'])[0](options,
                                                            params,
                                                            prefix='encoder_desc_sent',
                                                            nin=mult * options['dim'],
                                                            dim=options['dim'])
        if options['use_bidir']:
            params = get_layer(options['encoder_desc_sent'])[0](options,
                                                                params,
                                                                prefix='encoder_desc_sent_r',
                                                                nin=mult * options['dim'],
                                                                dim=options['dim'])
    ctxdim = mult * options['dim']
    logger.info("context dimensions is %d" % ctxdim)
    params = get_layer('ff')[0](options, params,
                                prefix='ff_att_ctx',
                                nin=ctxdim,
                                nout=options['dim'])

    # readout
    params = get_layer('ff')[0](options, params,
                                prefix='ff_att_q',
                                nin=ctxdim,
                                nout=options['dim'],
                                use_bias=False,
                                ortho=False)


    if options['use_desc_skip_c_g']:
        # readout for mean pooled desc
        params = get_layer('ff')[0](options, params,
                                    prefix='ff_out_mean_d',
                                    nin=ctxdim,
                                    nout=options['dim_word_ans'],
                                    use_bias=False,
                                    ortho=False)


    params = get_layer('ff')[0](options, params,
                                prefix='ff_out_q',
                                nin=ctxdim,
                                nout=options['dim_word_ans'],
                                ortho=False)

    params = get_layer('ff')[0](options, params,
                                prefix='ff_out_ctx',
                                nin=ctxdim,
                                nout=options['dim_word_ans'],
                                use_bias=False,
                                ortho=False)

    params = get_layer('ff')[0](options, params,
                                prefix='ff_logit',
                                nin=options['dim_word_ans'],
                                nout=options['n_words_ans'],
                                ortho=False)
    return params


def build_bidir_model(inp,
                      inp_mask,
                      tparams,
                      options,
                      sfx=None,
                      nsteps=None,
                      use_dropout=False,
                      use_noise=None,
                      truncate=None,
                      name=None):

    if use_dropout:
        assert use_noise is not None

    assert name is not None
    assert sfx is not None

    #inpr = inp[::-1]
    inpr_mask = inp_mask[::-1]

    n_timesteps = inp.shape[0]
    n_samples = inp.shape[1]

    emb = dot(inp, tparams['Wemb_%s' % sfx])
    emb = emb.reshape([n_timesteps, n_samples, -1])

    if use_dropout:
        emb = dropout_layer(emb, use_noise,
                            p=options['dropout_rate'])

    """
    Forward RNN
    """
    proj = get_layer(options[name])[1](tparams=tparams,
                                       state_below=emb,
                                       options=options,
                                       prefix=name,
                                       nsteps=nsteps,
                                       truncate=truncate,
                                       mask=inp_mask)

    """
    Reverse RNN.
    """
    #embr = dot(inpr, tparams['Wemb_%s' % sfx])
    embr = emb[::-1]#embr.reshape([n_timesteps, n_samples, -1])
    projr = get_layer(options[name])[1](tparams=tparams,
                                        state_below=embr,
                                        options=options,
                                        prefix=name + "_r",
                                        nsteps=nsteps,
                                        truncate=truncate,
                                        mask=inpr_mask)
    return proj, projr


def build_nonbidir_model(inp,
                         inp_mask,
                         tparams,
                         options,
                         sfx=None,
                         nsteps=None,
                         use_dropout=False,
                         use_noise=None,
                         truncate=None,
                         name=None):

    if use_dropout:
        assert use_noise is not None

    assert name is not None
    assert sfx is not None

    n_timesteps = inp.shape[0]
    n_samples = inp.shape[1]

    emb = dot(inp, tparams['Wemb_%s' % sfx])
    emb = emb.reshape([n_timesteps, n_samples, -1])

    if use_dropout:
        emb = dropout_layer(emb,
                            use_noise,
                            p=options['dropout_rate'])

    """
    Forward RNN
    """
    proj = get_layer(options[name])[1](tparams=tparams,
                                       state_below=emb,
                                       options=options,
                                       prefix=name,
                                       nsteps=nsteps,
                                       truncate=truncate,
                                       mask=inp_mask)
    return proj[0]


def build_attention(tparams,
                    options,
                    desc,
                    desc_mask,
                    dlen,
                    q,
                    q_mask=None,
                    sfx=None,
                    name=None):

    if desc.ndim != desc_mask.ndim:
        desc_mask_ = desc_mask.dimshuffle(0, 1, 'x')

    assert desc.ndim == desc_mask_.ndim

    if q_mask is not None:
        assert q.ndim == q_mask.ndim
        q *= q_mask

    masked_desc = desc * desc_mask_

    desc_in = desc.reshape((-1, desc.shape[-1]))
    projd = get_layer('ff')[1](tparams=tparams,
                               state_below=desc_in,
                               options=options,
                               prefix='ff_att_ctx',
                               activ='Linear')

    projq = get_layer('ff')[1](tparams, q,
                               options,
                               prefix='ff_att_q',
                               use_bias=False,
                               activ='Linear')

    """
    Unnormalized dist metric between the rep of desc and q.
    """
    sim_vals = 0
    if options['use_dq_sims']:
        q_proj = dot(q, tparams['ff_att_bi_dq'])
        desc_proj = dot(masked_desc,
                        tparams['ff_att_bi_dq']).reshape((masked_desc.shape[0],
                        masked_desc.shape[1], -1))
        sim_vals = (desc_proj * q_proj.dimshuffle('x', 0, 1)).sum(-1)
        sim_vals = sim_vals.dimshuffle(0, 1, 'x')

    projd = projd.reshape((masked_desc.shape[0], masked_desc.shape[1], -1))

    #Intermediate layer for annotation values.
    proj_att = Tanh(projd + projq.dimshuffle('x', 0, 1) + sim_vals)
    W_proj = tparams['ff_att_proj'].dimshuffle('x', 'x', 0)
    dot_proj = (W_proj * proj_att).sum(-1)
    pre_softmax = dot_proj
    alphas = Masked_Softmax(pre_softmax, mask=desc_mask, ax=0).dimshuffle(0, 1, 'x')
    ctx = (masked_desc * alphas).sum(0)

    return ctx, alphas


# build a training model
def build_model(tparams,
                options,
                prepare_data_fn,
                valid=None,
                cost_mask=None):

    opt_ret = dict()

    trng = RandomStreams(1234)
    use_noise = theano.shared(numpy.float32(0.))

    # description string: #words x #samples, description:
    if options['use_sent_reps']:
        x = tensor.tensor3('desc', dtype='uint32')
        word_mask = tensor.tensor3('desc_mask', dtype='float32')
        sent_mask = tensor.cast(word_mask.sum(0) > 0, "float32")
        slen = tensor.scalar('slen', dtype='uint32')
    else:
        x = tensor.matrix('desc', dtype="uint32")
        word_mask = tensor.matrix('desc_mask', dtype='float32')

    q = tensor.matrix('q', dtype="uint32")
    q_mask = tensor.matrix('q_mask', dtype="float32")
    y = tensor.vector('ans', dtype='uint32')
    em = tensor.matrix('entity_mask', dtype="float32")

    wlen = tensor.scalar('wlen', dtype='uint32')
    qlen = tensor.scalar('qlen', dtype='uint32')
    if options['debug']:
        if valid.done:
            valid.reset()

        valid_d = next(valid)
        d_, q_, a_, em_ = valid_d[0], valid_d[1], valid_d[2], valid_d[3]

        if options['use_sent_reps']:
            d_, d_mask_, q_, q_mask_, wlen_, slen_, qlen_ = prepare_data_fn(d_, q_)
        else:
            d_, d_mask_, q_, q_mask_, wlen_, qlen_ = prepare_data_fn(d_, q_)

        print "Debugging is enabled."

        theano.config.compute_test_value = 'warn'
        x.tag.test_value = numpy.array(d_).astype("uint32")
        word_mask.tag.test_value = numpy.array(d_mask_).astype("float32")
        q.tag.test_value = numpy.array(q_).astype("uint32")
        q_mask.tag.test_value = numpy.array(q_mask_).astype("float32")
        y.tag.test_value = numpy.array(a_).astype("uint32")
        em.tag.test_value = numpy.array(em_).astype("float32")
        wlen.tag.test_value = numpy.array(wlen_).astype("uint32")
        qlen.tag.test_value = numpy.array(qlen_).astype("uint32")

        if options['use_sent_reps']:
            slen.tag.test_value = numpy.array(slen_).astype("uint32")
            sent_mask.tag.test_value = numpy.array(d_mask_.sum(0) > 0, dtype="float32")

    if x.ndim == 3:
        x_rshp = x.reshape((x.shape[0], x.shape[1]*x.shape[2]))
    else:
        x_rshp = x

    """
        Bidirectional for the description.
    """
    if options['use_bidir']:
        proj_wx, proj_wxr = build_bidir_model(x_rshp,
                                              word_mask,
                                              tparams,
                                              options,
                                              sfx="word",
                                              nsteps=wlen,
                                              truncate=options['truncate'],
                                              use_dropout=options['use_dropout'],
                                              use_noise=use_noise,
                                              name="encoder_desc_word")

        desc_wrep = concatenate([proj_wx[0],
                                proj_wxr[0][::-1]],
                                axis=-1)
    else:
        proj_wx = build_nonbidir_model(x_rshp,
                                       word_mask,
                                       tparams,
                                       options,
                                       sfx="word",
                                       nsteps=wlen,
                                       truncate=options['truncate'],
                                       use_dropout=options['use_dropout'],
                                       use_noise=use_noise,
                                       name="encoder_desc_word")
        desc_wrep = proj_wx

    if options['use_bidir']:
        if options['use_sent_reps']:
            desc_wrep = desc_wrep.reshape((x.shape[0],
                                           x.shape[1],
                                           x.shape[2],
                                           -1))

            mean_desc_wrep = ((desc_wrep * word_mask.dimshuffle(0, 1, 2, 'x')).sum(0) /
                (word_mask.sum(0).dimshuffle(0, 1, 'x') + 1e-8))

            proj_sx, proj_sxr = build_bidir_model(mean_desc_wrep,
                                                  sent_mask,
                                                  tparams,
                                                  options,
                                                  sfx="sent",
                                                  nsteps=slen,
                                                  truncate=options['truncate'],
                                                  name="encoder_desc_sent")

            proj_x, proj_xr = proj_sx, proj_sxr
            desc_mask = sent_mask.dimshuffle(0, 1, 'x')
        else:
            proj_x, proj_xr = proj_wx, proj_wxr
            desc_mask = word_mask.dimshuffle(0, 1, 'x')

        """
        Build question bidir RNN
        """
        proj_q, proj_qr = build_bidir_model(q, q_mask,
                                            tparams,
                                            options, sfx="word",
                                            nsteps=qlen,
                                            truncate=options['truncate'],
                                            use_dropout=options['use_dropout'],
                                            use_noise=use_noise,
                                            name="encoder_q")

        desc_rep = concatenate([proj_x[0],
                                proj_xr[0][::-1]],
                                axis=-1)

        q_rep = concatenate([proj_q[0][-1],
                            proj_qr[0][::-1][0]],
                            axis=-1)

    else:
        if options['use_sent_reps']:
            desc_wrep = desc_wrep.reshape((x.shape[0],
                                           x.shape[1],
                                           x.shape[2],
                                           -1))

            mean_desc_wrep = ((desc_wrep * word_mask.dimshuffle(0, 1, 2, 'x')).sum(0) /
                (word_mask.sum(0).dimshuffle(0, 1, 'x') + 1e-8))

            proj_sx = build_nonbidir_model(mean_desc_wrep,
                                           sent_mask,
                                           tparams,
                                           options,
                                           sfx="sent",
                                           nsteps=slen,
                                           truncate=options['truncate'],
                                           name="encoder_desc_sent")
            proj_x = proj_sx
            desc_mask = sent_mask.dimshuffle(0, 1, 'x')
        else:
            proj_x = proj_wx
            desc_mask = word_mask.dimshuffle(0, 1, 'x')
        """
        Build question bidir RNN
        """
        proj_q = build_nonbidir_model(q, q_mask,
                                            tparams,
                                            options, sfx="word",
                                            nsteps=qlen,
                                            truncate=options['truncate'],
                                            use_dropout=options['use_dropout'],
                                            use_noise=use_noise,
                                            name="encoder_q")

        desc_rep = proj_x
        q_rep = proj_q[-1]

    g_desc_ave = 0.

    if options['use_desc_skip_c_g']:
        desc_mean = (desc_rep * desc_mask).sum(0) / \
                tensor.cast(desc_mask.sum(0), 'float32')

        g_desc_ave = get_layer('ff')[1](tparams,
                                        desc_mean,
                                        options,
                                        prefix='ff_out_mean_d',
                                        use_bias=False,
                                        activ='Linear')

    desc_ctx, alphas = build_attention(tparams,
                                       options,
                                       desc_rep,
                                       sent_mask \
                                               if options['use_sent_reps'] else word_mask,
                                       slen \
                                               if options['use_sent_reps'] else wlen,
                                       q=q_rep)

    opt_ret['dec_alphas'] = alphas
    opt_ret['desc_ctx'] = desc_ctx

    g_ctx = get_layer('ff')[1](tparams,
                               desc_ctx,
                               options,
                               prefix='ff_out_ctx',
                               use_bias=False,
                               activ='Linear')

    g_q = get_layer('ff')[1](tparams,
                             q_rep,
                             options,
                             prefix='ff_out_q',
                             activ='Linear')

    if options['use_elu_g']:
        g_out = ELU(g_ctx + g_q + g_desc_ave)
    else:
        g_out = Tanh(g_ctx + g_q + g_desc_ave)

    if options['use_dropout']:
        g_out = dropout_layer(g_out, use_noise,
                              p=options['dropout_rate'])

    logit = get_layer('ff')[1](tparams,
                               g_out,
                               options,
                               prefix='ff_logit',
                               activ='Linear')

    probs = Softmax(logit)
    hinge_cost = multiclass_hinge_loss(probs, y)

    # compute the cost
    cost, errors, ent_errors, ent_derrors = nll_simple(y,
                                                       probs,
                                                       cost_ent_mask=cost_mask,
                                                       cost_ent_desc_mask=em)
    cost = cost #+ 1e-2 * hinge_cost
    #cost = hinge_cost
    vals = OrderedDict({'desc': x,
                        'word_mask': word_mask,
                        'q': q,
                        'q_mask': q_mask,
                        'ans': y,
                        'wlen': wlen,
                        'ent_mask': em,
                        'qlen': qlen})

    if options['use_sent_reps']:
        vals['slen'] = slen

    return trng, use_noise, vals, opt_ret, \
            cost, errors, ent_errors, ent_derrors, \
            probs


def eval_model(f_log_probs,
               prepare_data,
               options,
               iterator,
               verbose=False,
               use_sent_rep=False):
    """
    To evaluate the model, for each example in the evaluation-set
    we have to compute the probabilities.
    """

    print "Started the evaluation."

    probs, errors, costs, alphas, error_ents, error_dents = [], [], [], [], [], []
    n_done = 0
    for batch in iterator:
        d = batch[0]
        q = batch[1]
        a = batch[2]
        em = batch[3]
        n_done += len(d)
        if not use_sent_rep:
            d, d_mask, q, q_mask, dlen, qlen = prepare_data(d, q)
            outs = f_log_probs(d,
                                                           d_mask, q,
                                                           q_mask, a, dlen,
                                                           qlen, em)
        else:
            d, d_mask, q, q_mask, dlen, slen, \
                    qlen = prepare_data(d, q)
            outs = f_log_probs(d,
                                                           d_mask,
                                                           q,
                                                           q_mask,
                                                           a,
                                                           dlen,
                                                           slen,
                                                           qlen, em)

        if len(outs) == 4:
            pcosts, perrors, pprobs, palphas = list(outs)
            perror_ent = None
            perror_dent = None
        elif len(outs) == 5:
            pcosts, perrors, pprobs, palphas, perror_ent = list(outs)
            perror_dent = None
        elif len(outs) == 6:
            pcosts, perrors, pprobs, palphas, perror_ent, perror_dent = list(outs)


        if isinstance(pcosts.tolist(), list):
            for i, (cost, pp, err, palpha) in enumerate(zip(pcosts, pprobs, perrors, palphas)):
                probs.append(pp)
                errors.append(err)
                costs.append(cost)
                alphas.append(palpha)
                if perror_ent:
                    error_ents.append(perror_ent[i])
                if perror_dent:
                    error_dents.append(perror_dent[i])
        else:
            probs.append(pprobs)
            errors.append(perrors)
            costs.append(pcosts)
            alphas.append(palphas)
            if perror_ent:
                error_ents.append(perror_ent)
            if perror_dent:
                error_dents.append(perror_dent)

        if numpy.isnan(numpy.mean(probs)):
            import ipdb; ipdb.set_trace()

        if verbose:
            print >>sys.stderr, '%d samples computed' % (n_done)

    print >>sys.stderr, 'Eval is done. \n Predictions over %d samples are computed.' % (n_done)
    error_ent = -1
    if len(error_ents) > 0:
        error_ent = numpy.mean(error_ents)

    error_dent = -1
    if len(error_dents) > 0:
        error_dent = numpy.mean(error_dents)

    return numpy.array(costs), numpy.array(errors), numpy.array(probs), \
            numpy.array(alphas), error_ent, error_dent
