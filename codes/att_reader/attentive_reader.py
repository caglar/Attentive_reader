import cPickle as pkl
import numpy
import copy
import theano
from theano import tensor

import os
import time

from collections import defaultdict

from utils import zipp, unzip, itemlist, load_params, init_tparams, prfx, \
        reset_train_vals, negentropy

from model import init_params, build_model, eval_model
from training import adadelta, adam, rmsprop, sgd, get_norms
from rc_data_iter import load_data, PytablesRCDataIterator
from pkl_data_iterator import load_data as load_pkl_data
from core.utils import ensure_dir_exists, safe_grad, sharedX
from core.learning_rule import Adasecant, Adam, RMSPropMomentum, AdaDelta
from core.utils.nnet_utils import running_ave


profile = False


# batch preparation
def prepare_data(seqs_x, seqs_y):
    seqs_x = [s[:-1] for s in seqs_x]
    seqs_y = [s[:-1] for s in seqs_y]

    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]
    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('uint32')
    y = numpy.zeros((maxlen_y, n_samples)).astype('uint32')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx], idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx], idx] = 1.

    return x, x_mask, y, y_mask, maxlen_x, maxlen_y


# batch preparation
def prepare_data_sents(seqs_x, seqs_y):
    # x: a list of sentences
    lengths_s = []
    lengths_w = []
    for seq in seqs_x:
        lengths_s.append(len(seq))
        lengths_w_tmp = []
        for w in seq:
            lengths_w_tmp.append(len(w))
        lengths_w.append(lengths_w_tmp)

    lengths_y = [len(s) for s in seqs_y]
    n_samples = len(seqs_x)
    maxlen_s = numpy.max(lengths_s) + 1
    maxlen_w = numpy.max([lw for lws in lengths_w for lw in lws]) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_w, maxlen_s, n_samples)).astype('uint32')
    y = numpy.zeros((maxlen_y, n_samples)).astype('uint32')
    x_mask = numpy.zeros((maxlen_w, maxlen_s, n_samples)).astype('float32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype('float32')

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        for i in xrange(lengths_s[idx]):
            x[:lengths_w[idx][i], i, idx] = numpy.array(s_x[i], dtype='uint32')
            x_mask[:lengths_w[idx][i], i, idx] = 1.
        if x[:, :, idx].sum((0, 1)) <= 1:
            import ipdb; ipdb.set_trace()
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx], idx] = 1.

    return x, x_mask, y, y_mask, maxlen_w, maxlen_s, maxlen_y


def print_param_norms(params):
    nparams = unzip(params)
    print "Printing out the parameter norms."
    for np, tp in zip(nparams.values(), params.keys()):
        nval = numpy.sqrt((np**2).sum())
        print "Norm of param %s: %f" % (tp, nval)


"""
    Get the hyperparameters and call the functions to construct the
computational graph in theano and run the mainloop.
"""
def train(dim_word_desc=400,# word vector dimensionality
          dim_word_q=400,
          dim_word_ans=600,
          dim_proj=300,
          dim=400,# the number of LSTM units
          encoder_desc='lstm',
          encoder_desc_word='lstm',
          encoder_desc_sent='lstm',
          use_dq_sims=False,
          eyem=None,
          learn_h0=False,
          use_desc_skip_c_g=False,
          debug=False,
          encoder_q='lstm',
          patience=10,
          max_epochs=5000,
          dispFreq=100,
          decay_c=0.,
          alpha_c=0.,
          clip_c=-1.,
          lrate=0.01,
          n_words_q=49145,
          n_words_desc=115425,
          n_words_ans=409,
          pkl_train_files=None,
          pkl_valid_files=None,
          maxlen=2000, # maximum length of the description
          optimizer='rmsprop',
          batch_size=2,
          vocab=None,
          valid_batch_size=16,
          use_elu_g=False,
          saveto='model.npz',
          model_dir=None,
          ms_nlayers=3,
          validFreq=1000,
          saveFreq=1000, # save the parameters after every saveFreq updates
          datasets=[None],
          truncate=400,
          momentum=0.9,
          use_bidir=False,
          cost_mask=None,
          valid_datasets=['/u/yyu/stor/caglar/rc-data/cnn/cnn_test_data.h5',
                          '/u/yyu/stor/caglar/rc-data/cnn/cnn_valid_data.h5'],
          dropout_rate=0.5,
          use_dropout=True,
          reload_=True,
          **opt_ds):

    ensure_dir_exists(model_dir)
    mpath = os.path.join(model_dir, saveto)
    mpath_best = os.path.join(model_dir, prfx("best", saveto))
    mpath_last = os.path.join(model_dir, prfx("last", saveto))
    mpath_stats = os.path.join(model_dir, prfx("stats", saveto))

    # Model options
    model_options = locals().copy()
    model_options['use_sent_reps'] = opt_ds['use_sent_reps']
    stats = defaultdict(list)

    del model_options['eyem']
    del model_options['cost_mask']

    if cost_mask is not None:
        cost_mask = sharedX(cost_mask)

    # reload options and parameters
    if reload_:
        print "Reloading the model."
        if os.path.exists(mpath_best):
            print "Reloading the best model from %s." % mpath_best
            with open(os.path.join(mpath_best, '%s.pkl' % mpath_best), 'rb') as f:
                models_options = pkl.load(f)
            params = init_params(model_options)
            params = load_params(mpath_best, params)
        elif os.path.exists(mpath):
            print "Reloading the model from %s." % mpath
            with open(os.path.join(mpath, '%s.pkl' % mpath), 'rb') as f:
                models_options = pkl.load(f)
            params = init_params(model_options)
            params = load_params(mpath, params)
        else:
            raise IOError("Couldn't open the file.")
    else:
        print "Couldn't reload the models initializing from scratch."
        params = init_params(model_options)

    if datasets[0]:
        print "Short dataset", datasets[0]

    print 'Loading data'
    print 'Building model'
    if pkl_train_files is None or pkl_valid_files is None:
        train, valid, test = load_data(path=datasets[0],
                                       valid_path=valid_datasets[0],
                                       test_path=valid_datasets[1],
                                       batch_size=batch_size,
                                       **opt_ds)
    else:
        train, valid, test = load_pkl_data(train_file_paths=pkl_train_files,
                                           valid_file_paths=pkl_valid_files,
                                           batch_size=batch_size,
                                           vocab=vocab,
                                           eyem=eyem,
                                           **opt_ds)

    tparams = init_tparams(params)
    trng, use_noise, inps_d, \
                     opt_ret, \
                     cost, errors, ent_errors, ent_derrors, probs = \
                        build_model(tparams,
                                    model_options,
                                    prepare_data if not opt_ds['use_sent_reps'] \
                                            else prepare_data_sents,
                                    valid,
                                    cost_mask=cost_mask)

    alphas = opt_ret['dec_alphas']

    if opt_ds['use_sent_reps']:
        inps = [inps_d["desc"], \
                inps_d["word_mask"], \
                inps_d["q"], \
                inps_d['q_mask'], \
                inps_d['ans'], \
                inps_d['wlen'],
                inps_d['slen'], inps_d['qlen'],\
                inps_d['ent_mask']
                ]
    else:
        inps = [inps_d["desc"], \
                inps_d["word_mask"], \
                inps_d["q"], \
                inps_d['q_mask'], \
                inps_d['ans'], \
                inps_d['wlen'], \
                inps_d['qlen'], \
                inps_d['ent_mask']]

    outs = [cost, errors, probs, alphas]
    if ent_errors:
        outs += [ent_errors]

    if ent_derrors:
        outs += [ent_derrors]

    # before any regularizer
    print 'Building f_log_probs...',
    f_log_probs = theano.function(inps, outs, profile=profile)
    print 'Done'

    # Apply weight decay on the feed-forward connections
    if decay_c > 0.:
        decay_c = theano.shared(numpy.float32(decay_c), name='decay_c')
        weight_decay = 0.

        for kk, vv in tparams.iteritems():
            if "logit" in kk or "ff" in kk:
                weight_decay += (vv ** 2).sum()

        weight_decay *= decay_c
        cost += weight_decay

    # after any regularizer
    print 'Computing gradient...',
    grads = safe_grad(cost, itemlist(tparams))
    print 'Done'

    # Gradient clipping:
    if clip_c > 0.:
        g2 = get_norms(grads)
        for p, g in grads.iteritems():
            grads[p] = tensor.switch(g2 > (clip_c**2),
                                     (g / tensor.sqrt(g2 + 1e-8)) * clip_c,
                                     g)
    inps.pop()
    if optimizer.lower() == "adasecant":
        learning_rule = Adasecant(delta_clip=25.0,
                                  use_adagrad=True,
                                  grad_clip=0.25,
                                  gamma_clip=0.)
    elif optimizer.lower() == "rmsprop":
        learning_rule = RMSPropMomentum(init_momentum=momentum)
    elif optimizer.lower() == "adam":
        learning_rule = Adam()
    elif optimizer.lower() == "adadelta":
        learning_rule = AdaDelta()

    lr = tensor.scalar(name='lr')
    print 'Building optimizers...',
    learning_rule = None

    if learning_rule:
        f_grad_shared, f_update = learning_rule.get_funcs(learning_rate=lr,
                                                          grads=grads,
                                                          inp=inps,
                                                          cost=cost,
                                                          errors=errors)
    else:
        f_grad_shared, f_update = eval(optimizer)(lr,
                                                  tparams,
                                                  grads,
                                                  inps,
                                                  cost,
                                                  errors)

    print 'Done'
    print 'Optimization'
    history_errs = []
    # reload history
    if reload_ and os.path.exists(mpath):
        history_errs = list(numpy.load(mpath)['history_errs'])

    best_p = None
    bad_count = 0

    if validFreq == -1:
        validFreq = len(train[0]) / batch_size

    if saveFreq == -1:
        saveFreq = len(train[0]) / batch_size

    best_found = False
    uidx = 0
    estop = False

    train_cost_ave, train_err_ave, \
            train_gnorm_ave = reset_train_vals()

    for eidx in xrange(max_epochs):
        n_samples = 0

        if train.done:
            train.reset()

        for d_, q_, a, em in train:
            n_samples += len(a)
            uidx += 1
            use_noise.set_value(1.)

            if opt_ds['use_sent_reps']:
                # To mask the description and the question.
                d, d_mask, q, q_mask, dlen, slen, qlen = prepare_data_sents(d_,
                                                                            q_)

                if d is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    uidx -= 1
                    continue

                ud_start = time.time()
                cost, errors, gnorm, pnorm = f_grad_shared(d,
                                                           d_mask,
                                                           q,
                                                           q_mask,
                                                           a,
                                                           dlen,
                                                           slen,
                                                           qlen)
            else:
                d, d_mask, q, q_mask, dlen, qlen = prepare_data(d_, q_)

                if d is None:
                    print 'Minibatch with zero sample under length ', maxlen
                    uidx -= 1
                    continue

                ud_start = time.time()
                cost, errors, gnorm, pnorm = f_grad_shared(d, d_mask,
                                                           q, q_mask,
                                                           a,
                                                           dlen,
                                                           qlen)

            upnorm = f_update(lrate)
            ud = time.time() - ud_start

            # Collect the running ave train stats.
            train_cost_ave = running_ave(train_cost_ave,
                                         cost)
            train_err_ave = running_ave(train_err_ave,
                                        errors)
            train_gnorm_ave = running_ave(train_gnorm_ave,
                                          gnorm)

            if numpy.isnan(cost) or numpy.isinf(cost):
                print 'NaN detected'
                import ipdb; ipdb.set_trace()

            if numpy.mod(uidx, dispFreq) == 0:
                print 'Epoch ', eidx, ' Update ', uidx, \
                        ' Cost ', cost, ' UD ', ud, \
                        ' UpNorm ', upnorm[0].tolist(), \
                        ' GNorm ', gnorm, \
                        ' Pnorm ', pnorm, 'Terrors ', errors

            if numpy.mod(uidx, saveFreq) == 0:
                print 'Saving...',
                if best_p is not None and best_found:
                    numpy.savez(mpath_best, history_errs=history_errs, **best_p)
                    pkl.dump(model_options, open('%s.pkl' % mpath_best, 'wb'))
                else:
                    params = unzip(tparams)

                numpy.savez(mpath, history_errs=history_errs, **params)
                pkl.dump(model_options, open('%s.pkl' % mpath, 'wb'))
                pkl.dump(stats, open("%s.pkl" % mpath_stats, 'wb'))

                print 'Done'
                print_param_norms(tparams)

            if numpy.mod(uidx, validFreq) == 0:
                use_noise.set_value(0.)
                if valid.done:
                    valid.reset()

                valid_costs, valid_errs, valid_probs, \
                        valid_alphas, error_ent, error_dent = eval_model(f_log_probs,
                                                  prepare_data if not opt_ds['use_sent_reps'] \
                                                    else prepare_data_sents,
                                                  model_options,
                                                  valid,
                                                  use_sent_rep=opt_ds['use_sent_reps'])

                valid_alphas_ = numpy.concatenate([va.argmax(0) for va  in valid_alphas.tolist()], axis=0)
                valid_err = valid_errs.mean()
                valid_cost = valid_costs.mean()
                valid_alpha_ent = -negentropy(valid_alphas)

                mean_valid_alphas = valid_alphas_.mean()
                std_valid_alphas = valid_alphas_.std()

                mean_valid_probs = valid_probs.argmax(1).mean()
                std_valid_probs = valid_probs.argmax(1).std()

                history_errs.append([valid_cost, valid_err])

                stats['train_err_ave'].append(train_err_ave)
                stats['train_cost_ave'].append(train_cost_ave)
                stats['train_gnorm_ave'].append(train_gnorm_ave)

                stats['valid_errs'].append(valid_err)
                stats['valid_costs'].append(valid_cost)
                stats['valid_err_ent'].append(error_ent)
                stats['valid_err_desc_ent'].append(error_dent)

                stats['valid_alphas_mean'].append(mean_valid_alphas)
                stats['valid_alphas_std'].append(std_valid_alphas)
                stats['valid_alphas_ent'].append(valid_alpha_ent)

                stats['valid_probs_mean'].append(mean_valid_probs)
                stats['valid_probs_std'].append(std_valid_probs)

                if uidx == 0 or valid_err <= numpy.array(history_errs)[:, 1].min():
                    best_p = unzip(tparams)
                    bad_counter = 0
                    best_found = True
                else:
                    bst_found = False

                if numpy.isnan(valid_err):
                    import ipdb; ipdb.set_trace()


                print "============================"
                print '\t>>>Valid error: ', valid_err, \
                        ' Valid cost: ', valid_cost
                print '\t>>>Valid pred mean: ', mean_valid_probs, \
                        ' Valid pred std: ', std_valid_probs
                print '\t>>>Valid alphas mean: ', mean_valid_alphas, \
                        ' Valid alphas std: ', std_valid_alphas, \
                        ' Valid alpha negent: ', valid_alpha_ent, \
                        ' Valid error ent: ', error_ent, \
                        ' Valid error desc ent: ', error_dent

                print "============================"
                print "Running average train stats "
                print '\t>>>Train error: ', train_err_ave, \
                        ' Train cost: ', train_cost_ave, \
                        ' Train grad norm: ', train_gnorm_ave
                print "============================"


                train_cost_ave, train_err_ave, \
                    train_gnorm_ave = reset_train_vals()


        print 'Seen %d samples' % n_samples

        if estop:
            break

    if best_p is not None:
        zipp(best_p, tparams)

    use_noise.set_value(0.)
    valid.reset()
    valid_cost, valid_error, valid_probs, \
            valid_alphas, error_ent = eval_model(f_log_probs,
                                      prepare_data if not opt_ds['use_sent_reps'] \
                                           else prepare_data_sents,
                                      model_options, valid,
                                      use_sent_rep=opt_ds['use_sent_rep'])

    print " Final eval resuts: "
    print 'Valid error: ', valid_error.mean()
    print 'Valid cost: ', valid_cost.mean()
    print '\t>>>Valid pred mean: ', valid_probs.mean(), \
            ' Valid pred std: ', valid_probs.std(), \
            ' Valid error ent: ', error_ent

    params = copy.copy(best_p)

    numpy.savez(mpath_last,
                zipped_params=best_p,
                history_errs=history_errs,
                **params)

    return valid_err, valid_cost
