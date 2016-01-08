__authors__ = ["Caglar Gulcehre", "Junyoung Chung", "Laurent Dinh"]
__copyright__ = "Copyright 2014, Universite de Montreal"
__credits__ = "Laurent Dinh"
__license__ = "3-clause BSD"
__maintainer__ = "Junyoung Chung"
__email__ = "chungjun@iro"

import theano

import numpy as np
from theano import config
from theano import tensor as T
import logging

from theano.compat.python2x import OrderedDict
from core.utils import sharedX, as_floatX

logger = logging.getLogger(__name__)


class LearningRule():

    def add_channels_to_monitor(self, monitor, monitoring_dataset):
        """
        Method called by the training algorithm, which allows LearningRules to
        add monitoring channels.

        Parameters
        ----------
        monitor : pylearn2.monitor.Monitor
            Monitor object, to which the rule should register additional
            monitoring channels.
        monitoring_dataset : pylearn2.datasets.dataset.Dataset or dict
            Dataset instance or dictionary whose values are Dataset objects.
        """
        pass

    def get_updates(self, learning_rate, grads, lr_scalers=None):
        """
        Provides the symbolic (theano) description of the updates needed to
        perform this learning rule.

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.

        Returns
        -------
        updates : OrderdDict
            A dictionary mapping from the old model parameters, to their new
            values after a single iteration of the learning rule.

        Notes
        -----
        e.g. for standard SGD, one would return `sgd_rule_updates` defined
        below. Note that such a `LearningRule` object is not implemented, as
        these updates are implemented by default when the `learning_rule`
        parameter of sgd.SGD.__init__ is None.

        .. code-block::  python

            sgd_rule_updates = OrderedDict()
            for (param, grad) in grads.iteritems():
                sgd_rule_updates[k] = (param - learning_rate *
                                       lr_scalers.get(param, 1.) * grad)
        """
        raise NotImplementedError(str(type(self)) + " does not implement "
                                  "get_updates.")


class Momentum(LearningRule):
    """
    Implements momentum as described in Section 9 of
    "A Practical Guide to Training Restricted Boltzmann Machines",
    Geoffrey Hinton.

    Parameters are updated by the formula:
    inc := momentum * inc - learning_rate * d cost / d param
    param := param + inc

    Parameters
    ----------
    init_momentum : float
        Initial value for the momentum coefficient. It remains fixed during
        training unless used with a `MomentumAdjustor`
        extension.
    nesterov_momentum: bool
        Use the accelerated momentum technique described in:
        "Advances in Optimizing Recurrent Networks", Yoshua Bengio, et al.

    """

    def __init__(self, init_momentum, nesterov_momentum=False):
        assert init_momentum >= 0.
        assert init_momentum < 1.
        self.momentum = sharedX(init_momentum, 'momentum')
        self.nesterov_momentum = nesterov_momentum

    def get_funcs(self, learning_rate, grads, inp, cost, errors, lr_scalers=None):
        """
        Provides the updates for learning with gradient descent + momentum.

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """
        gshared = OrderedDict({p: sharedX(p.get_value() * 0.,
                             name='%s_grad' % p.name)
                             for p, g in grads.iteritems()})

        gsup = [(gs, g) for gs, g in zip(gshared.values(), grads.values())]
        get_norms = lambda x: T.sqrt(sum(map(lambda y: (y**2).sum(), x)))
        gnorm = get_norms(grads.values())
        pnorm = get_norms(grads.keys())
        f_grad_shared = theano.function(inp,
                                        [cost, errors, gnorm, pnorm],
                                        updates=gsup)
        updates = OrderedDict()

        for param, grad in gshared.keys():
            vel = sharedX(param.get_value() * 0.)
            assert param.dtype == vel.dtype
            assert grad.dtype == param.dtype
            if param.name is not None:
                vel.name = 'vel_' + param.name

            scaled_lr = learning_rate * lr_scalers.get(param, 1.)
            updates[vel] = self.momentum * vel - scaled_lr * grad

            inc = updates[vel]
            if self.nesterov_momentum:
                inc = self.momentum * inc - scaled_lr * grad

            assert inc.dtype == vel.dtype
            updates[param] = param + inc

        f_update = theano.function([learning_rate],
                                   [],
                                   updates=updates,
                                   on_unused_input='ignore')

        return f_grad_shared, f_update


class AdaDelta(LearningRule):
    """
    Implements the AdaDelta learning rule as described in:
    "AdaDelta: An Adaptive Learning Rate Method", Matthew D. Zeiler.

    Parameters
    ----------
    decay : float, optional
        Decay rate :math:`\\rho` in Algorithm 1 of the aforementioned
        paper.
    """

    def __init__(self, decay=0.95):
        assert decay >= 0.
        assert decay < 1.
        self.decay = decay

    def get_funcs(self, learning_rate, grads, inp, cost, errors, lr_scalers=None):
        """
        Compute the AdaDelta updates

        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """
        updates = OrderedDict()

        tot_norm_up = 0

        gshared = OrderedDict({p: sharedX(p.get_value() * 0.,
                             name='%s_grad' % p.name)
                             for p, g in grads.iteritems()})

        gsup = [(gshared[p], g) for p, g in grads.iteritems()]
        get_norms = lambda x: T.sqrt(sum(map(lambda y: (y**2).sum(), x)))
        gnorm = get_norms(grads.values())
        pnorm = get_norms(grads.keys())
        f_grad_shared = theano.function(inp,
                                        [cost, errors, gnorm, pnorm],
                                        updates=gsup)

        for param in gshared.keys():
            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(param.get_value() * 0.)
            # mean_square_dx := E[(\Delta x)^2]_{t-1}
            mean_square_dx = sharedX(param.get_value() * 0.)

            if param.name is not None:
                mean_square_grad.name = 'mean_square_grad_' + param.name
                mean_square_dx.name = 'mean_square_dx_' + param.name

            # Accumulate gradient
            new_mean_squared_grad = (
                self.decay * mean_square_grad +
                (1 - self.decay) * T.sqr(gshared[param])
            )

            # Compute update
            epsilon = learning_rate
            rms_dx_tm1 = T.sqrt(mean_square_dx + epsilon)
            rms_grad_t = T.sqrt(new_mean_squared_grad + epsilon)
            delta_x_t = - rms_dx_tm1 / rms_grad_t * gshared[param]

            # Accumulate updates
            new_mean_square_dx = (
                self.decay * mean_square_dx +
                (1 - self.decay) * T.sqr(delta_x_t)
            )

            # Apply update
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[param] = param + delta_x_t

            tot_norm_up += delta_x_t.norm(2)

        f_update = theano.function([learning_rate], [tot_norm_up],
                                   updates=updates,
                                   on_unused_input='ignore')

        return f_grad_shared, f_update


class RMSPropMomentum(Momentum):
    """
    Implements the RMSprop

    Parameters
    ----------
    """

    def __init__(self,
                 init_momentum,
                 averaging_coeff=0.95,
                 stabilizer=1e-2,
                 use_first_order=False,
                 bound_inc=False,
                 momentum_clipping=None):
        init_momentum = float(init_momentum)
        assert init_momentum >= 0.
        assert init_momentum <= 1.
        averaging_coeff = float(averaging_coeff)
        assert averaging_coeff >= 0.
        assert averaging_coeff <= 1.
        stabilizer = float(stabilizer)
        assert stabilizer >= 0.

        self.__dict__.update(locals())
        del self.self
        self.momentum = sharedX(self.init_momentum)

        self.momentum_clipping = momentum_clipping
        if momentum_clipping is not None:
            self.momentum_clipping = np.cast[config.floatX](momentum_clipping)

    def get_funcs(self, learning_rate, grads, inp, cost, errors, lr_scalers=None):
        """
        .. todo::

            WRITEME
        """
        updates = OrderedDict()
        velocity = OrderedDict()
        tot_norm_up = 0

        gshared = OrderedDict({p: sharedX(p.get_value() * 0.,
                             name='%s_grad' % p.name)
                             for p, g in grads.iteritems()})

        gsup = [(gshared[p], g) for p, g in grads.iteritems()]
        get_norms = lambda x: T.sqrt(sum(map(lambda y: (y**2).sum(), x)))
        gnorm = get_norms(grads.values())
        pnorm = get_norms(grads.keys())
        f_grad_shared = theano.function(inp,
                                        [cost, errors, gnorm, pnorm],
                                        updates=gsup)

        for param in gshared.keys():
            avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))
            velocity[param] = sharedX(np.zeros_like(param.get_value()))

            if param.name is not None:
                avg_grad_sqr.name = 'avg_grad_sqr_' + param.name

            new_avg_grad_sqr = self.averaging_coeff * avg_grad_sqr +\
                (1 - self.averaging_coeff) * T.sqr(gshared[param])
            if self.use_first_order:
                avg_grad = sharedX(np.zeros_like(param.get_value()))
                if param.name is not None:
                    avg_grad.name = 'avg_grad_' + param.name
                new_avg_grad = self.averaging_coeff * avg_grad +\
                    (1 - self.averaging_coeff) * gshared[param]
                rms_grad_t = T.sqrt(new_avg_grad_sqr - new_avg_grad**2)
                updates[avg_grad] = new_avg_grad
            else:
                rms_grad_t = T.sqrt(new_avg_grad_sqr)

            rms_grad_t = T.maximum(rms_grad_t, self.stabilizer)
            normalized_grad = gshared[param] / (rms_grad_t)
            new_velocity = self.momentum * velocity[param] -\
                learning_rate * normalized_grad
            tot_norm_up += new_velocity.norm(2)

            updates[avg_grad_sqr] = new_avg_grad_sqr
            updates[velocity[param]] = new_velocity
            updates[param] = param + new_velocity

        if self.momentum_clipping is not None:
            tot_norm_up = 0

            new_mom_norm = sum(
                map(lambda X: T.sqr(X).sum(),
                    [updates[velocity[param]] for param in grads.keys()])
            )
            new_mom_norm = T.sqrt(new_mom_norm)
            scaling_den = T.maximum(self.momentum_clipping, new_mom_norm)
            scaling_num = self.momentum_clipping

            for param in grads.keys():
                if self.bound_inc:
                    updates[velocity[param]] *= (scaling_num / scaling_den)
                    updates[param] = param + updates[velocity[param]]
                else:
                    update_step = updates[velocity[param]] * (scaling_num / scaling_den)
                    tot_norm_up += update_step.norm(2)
                    updates[param] = param + update_step

        f_update = theano.function([learning_rate], [tot_norm_up],
                                   updates=updates,
                                   on_unused_input='ignore')

        return f_grad_shared, f_update


class Adam(Momentum):
    """
    The MIT License (MIT)

    Copyright (c) 2015 Alec Radford

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    """
    def __init__(self,
                 init_momentum=0.9,
                 averaging_coeff=0.99,
                 stabilizer=1e-4,
                 update_param_norm_ratio=0.003,
                 gradient_clipping=None):
        init_momentum = float(init_momentum)
        assert init_momentum >= 0.
        assert init_momentum <= 1.
        averaging_coeff = float(averaging_coeff)
        assert averaging_coeff >= 0.
        assert averaging_coeff <= 1.
        stabilizer = float(stabilizer)
        assert stabilizer >= 0.

        self.__dict__.update(locals())
        del self.self
        self.momentum = sharedX(self.init_momentum)
        self.update_param_norm_ratio = update_param_norm_ratio

        self.gradient_clipping = gradient_clipping
        if gradient_clipping is not None:
            self.gradient_clipping = np.cast[config.floatX](gradient_clipping)

    def get_funcs(self, learning_rate, grads, inp, cost, errors, lr_scalers=None):
        """
        .. todo::

            WRITEME
        """
        if self.gradient_clipping is not None:
            grads_norm = sum(
                map(lambda X: T.sqr(X).sum(),
                    [grads[param] for param in grads.keys()])
            )
            grads_norm = T.sqrt(grads_norm)
            scaling_den = T.maximum(self.gradient_clipping, grads_norm)
            scaling_num = self.gradient_clipping
            for param in grads.keys():
                grads[param] = scaling_num * grads[param] / scaling_den

        updates = OrderedDict()
        velocity = OrderedDict()
        normalized_velocities = OrderedDict()

        counter = sharedX(0, 'counter')
        tot_norm_up = 0
        gshared = OrderedDict({p: sharedX(p.get_value() * 0.,
                             name='%s_grad' % p.name)
                             for p, g in grads.iteritems()})

        gsup = [(gshared[p], g) for p, g in grads.iteritems()]
        get_norms = lambda x: T.sqrt(sum(map(lambda y: (y**2).sum(), x)))
        gnorm = get_norms(grads.values())
        pnorm = get_norms(grads.keys())
        f_grad_shared = theano.function(inp,
                                        [cost, errors, gnorm, pnorm],
                                        updates=gsup)
        for param in gshared.keys():
            avg_grad_sqr = sharedX(np.zeros_like(param.get_value()))
            velocity[param] = sharedX(np.zeros_like(param.get_value()))

            next_counter = counter + 1.

            fix_first_moment = 1. - self.momentum**next_counter
            fix_second_moment = 1. - self.averaging_coeff**next_counter

            if param.name is not None:
                avg_grad_sqr.name = 'avg_grad_sqr_' + param.name

            new_avg_grad_sqr = self.averaging_coeff*avg_grad_sqr \
                + (1 - self.averaging_coeff)*T.sqr(gshared[param])

            rms_grad_t = T.sqrt(new_avg_grad_sqr)
            rms_grad_t = T.maximum(rms_grad_t, self.stabilizer)
            new_velocity = self.momentum * velocity[param] \
                - (1 - self.momentum) * gshared[param]
            normalized_velocity = (new_velocity * T.sqrt(fix_second_moment)) \
                / (rms_grad_t * fix_first_moment)

            tot_norm_up += learning_rate*normalized_velocity.norm(2)

            normalized_velocities[param] = normalized_velocity
            updates[avg_grad_sqr] = new_avg_grad_sqr
            updates[velocity[param]] = new_velocity
            updates[param] = param + normalized_velocities[param]

        updates[counter] = counter + 1
        f_update = theano.function([learning_rate], [tot_norm_up],
                                   updates=updates,
                                   on_unused_input='ignore')

        return f_grad_shared, f_update


class Adasecant(LearningRule):
    """
    Adasecant:
        Based on the paper:
            Gulcehre, Caglar, and Yoshua Bengio.
            "ADASECANT: Robust Adaptive Secant Method for Stochastic Gradient."
            arXiv preprint arXiv:1412.7419 (2014).
    There are some small changes in this code.
    Parameters
    ----------

    gamma_clip : float, optional
        The clipping threshold for the gamma. In general 1.8 seems to
        work fine for several tasks.
    decay : float, optional
        Decay rate :math:`\\rho` in Algorithm 1 of the aforementioned
        paper. Decay 0.95 seems to work fine for several tasks.
    start_var_reduction: float, optional,
        How many updates later should the variance reduction start from?
    delta_clip: float, optional,
        The threshold to clip the deltas after.
    grad_clip: float, optional,
        Apply gradient clipping for RNNs (not necessary for feedforward networks). But this is
        a constraint on the norm of the gradient per layer.
        Based on:
            Pascanu, Razvan, Tomas Mikolov, and Yoshua Bengio. "On the difficulty of training
            recurrent neural networks." arXiv preprint arXiv:1211.5063 (2012).
    use_adagrad: bool, optional
        Either to use clipped adagrad or not.
    use_corrected_grad: bool, optional
        Either to use correction for gradients (referred as variance
        reduction in the workshop paper).
    """
    def __init__(self, decay=0.95,
                 gamma_clip=0.0,
                 grad_clip=None,
                 start_var_reduction=0,
                 delta_clip=None,
                 gamma_reg=1e-6,
                 slow_decay=0.995,
                 learning_rate=1.0,
                 use_adagrad=False,
                 perform_update=True,
                 skip_nan_inf=False,
                 use_corrected_grad=True):

        assert decay >= 0.
        assert decay < 1.

        self.start_var_reduction = start_var_reduction
        self.delta_clip = delta_clip
        self.gamma_clip = gamma_clip
        self.grad_clip = grad_clip
        self.slow_decay = slow_decay
        self.decay = sharedX(decay, "decay")
        self.use_corrected_grad = use_corrected_grad
        self.use_adagrad = use_adagrad
        self.gamma_reg = gamma_reg
        self.damping = 1e-7
        self.learning_rate = learning_rate
        self.perform_update = perform_update

        # We have to bound the tau to prevent it to
        # grow to an arbitrarily large number, oftenwise
        # that causes numerical instabilities for very deep
        # networks. Note that once tau become very large, it will keep,
        # increasing indefinitely.
        self.skip_nan_inf = skip_nan_inf
        self.upper_bound_tau = 1e7
        self.lower_bound_tau = 1.5

    def get_funcs(self, learning_rate, grads, inp, cost, errors, lr_scalers=None):
        """
        .. todo::
            WRITEME
        Parameters
        ----------
        learning_rate : float
            Learning rate coefficient. Learning rate is not being used but, pylearn2 requires a
            learning rate to be defined.
        grads : dict
            A dictionary mapping from the model's parameters to their
            gradients.
        lr_scalers : dict
            A dictionary mapping from the model's parameters to a learning
            rate multiplier.
        """

        updates = OrderedDict({})
        eps = self.damping
        step = sharedX(0., name="step")

        if self.skip_nan_inf:
            #If norm of the gradients of a parameter is inf or nan don't update that parameter
            #That might be useful for RNNs.
            grads = OrderedDict({p: T.switch(T.or_(T.isinf(grads[p]),
                T.isnan(grads[p])), 0, grads[p]) for
                p in grads.keys()})

        # Block-normalize gradients:
        nparams = len(grads.keys())

        # Apply the gradient clipping, this is only sometimes
        # necessary for RNNs and sometimes for very deep networks
        if self.grad_clip:
            assert self.grad_clip > 0.
            assert self.grad_clip <= 1., "Norm of the gradients per layer can not be larger than 1."

            gnorm = sum([g.norm(2) for g in grads.values()])
            notfinite = T.or_(T.isnan(gnorm), T.isinf(gnorm))

            for p, g in grads.iteritems():
                tmpg = T.switch(gnorm / nparams > self.grad_clip,
                                 g * self.grad_clip * nparams / gnorm , g)
                grads[p] = T.switch(notfinite, as_floatX(0.1)*p, tmpg)

        tot_norm_up = 0
        gshared = OrderedDict({p: sharedX(p.get_value() * 0.,
                             name='%s_grad' % p.name)
                             for p, g in grads.iteritems()})

        gsup = [(gshared[p], g) for p, g in grads.iteritems()]
        get_norms = lambda x: T.sqrt(sum(map(lambda y: (y**2).sum(), x)))
        gnorm = get_norms(grads.values())
        pnorm = get_norms(grads.keys())
        f_grad_shared = theano.function(inp,
                                        [cost, errors, gnorm, pnorm],
                                        updates=gsup)

        fix_decay = self.slow_decay**(step + 1)

        for param in gshared.keys():
            gshared[param].name = "grad_%s" % param.name
            mean_grad = sharedX(param.get_value() * 0. + eps, name="mean_grad_%s" % param.name)
            gnorm_sqr = sharedX(0.0 + eps, name="gnorm_%s" % param.name)

            prod_taus = sharedX((np.ones_like(param.get_value()) - 2*eps),
                                 name="prod_taus_x_t_" + param.name)
            slow_constant = 2.1

            if self.use_adagrad:
                # sum_square_grad := \sum_i g_i^2
                sum_square_grad = sharedX(param.get_value(borrow=True) * 0.,
                                          name="sum_square_grad_%s" % param.name)

            """
               Initialization of accumulators
            """
            taus_x_t = sharedX((np.ones_like(param.get_value()) + eps) * slow_constant,
                               name="taus_x_t_" + param.name)
            self.taus_x_t = taus_x_t

            #Variance reduction parameters
            #Numerator of the gamma:
            gamma_nume_sqr = sharedX(np.zeros_like(param.get_value()) + eps,
                                     name="gamma_nume_sqr_" + param.name)

            #Denominator of the gamma:
            gamma_deno_sqr = sharedX(np.zeros_like(param.get_value()) + eps,
                                     name="gamma_deno_sqr_" + param.name)

            #For the covariance parameter := E[\gamma \alpha]_{t-1}
            cov_num_t = sharedX(np.zeros_like(param.get_value()) + eps,
                                name="cov_num_t_" + param.name)

            # mean_squared_grad := E[g^2]_{t-1}
            mean_square_grad = sharedX(np.zeros_like(param.get_value()) + eps,
                                       name="msg_" + param.name)

            # mean_square_dx := E[(\Delta x)^2]_{t-1}
            mean_square_dx = sharedX(param.get_value() * 0., name="msd_" + param.name)

            if self.use_corrected_grad:
                old_grad = sharedX(param.get_value() * 0. + eps)

            #The uncorrected gradient of previous of the previous update:
            old_plain_grad = sharedX(param.get_value() * 0. + eps)
            mean_curvature = sharedX(param.get_value() * 0. + eps)
            mean_curvature_sqr = sharedX(param.get_value() * 0. + eps)

            # Initialize the E[\Delta]_{t-1}
            mean_dx = sharedX(param.get_value() * 0.)

            # Block-wise normalize the gradient:
            norm_grad = gshared[param]

            #For the first time-step, assume that delta_x_t := norm_grad
            gnorm = T.sqr(norm_grad).sum()

            cond = T.eq(step, 0)
            gnorm_sqr_o = cond * gnorm + (1 - cond) * gnorm_sqr
            gnorm_sqr_b = gnorm_sqr_o / (1 - fix_decay)

            norm_grad = norm_grad / (T.sqrt(gnorm_sqr_b) + eps)
            msdx = cond * norm_grad**2 + (1 - cond) * mean_square_dx
            mdx = cond * norm_grad + (1 - cond) * mean_dx

            new_prod_taus = (
                prod_taus * (1 - 1 / taus_x_t)
            )

            """
                Compute the new updated values.
            """
            # E[g_i^2]_t
            new_mean_squared_grad = (
                mean_square_grad * (1 - 1 / taus_x_t)  +
                T.sqr(norm_grad) / (taus_x_t)
            )
            new_mean_squared_grad.name = "msg_" + param.name

            # E[g_i]_t
            new_mean_grad = (
                mean_grad * (1 - 1 / taus_x_t) +
                norm_grad / taus_x_t
            )

            new_mean_grad.name = "nmg_" + param.name
            mg = new_mean_grad / (1 - new_prod_taus)
            mgsq = new_mean_squared_grad / (1 - new_prod_taus)

            new_gnorm_sqr = (
                    gnorm_sqr_o * self.slow_decay +
                    T.sqr(norm_grad).sum() * (1 - self.slow_decay)
            )

            # Keep the rms for numerator and denominator of gamma.
            new_gamma_nume_sqr = (
                gamma_nume_sqr * (1 - 1 / taus_x_t) +
                T.sqr((norm_grad - old_grad) * (old_grad - mg)) / taus_x_t
            )
            new_gamma_nume_sqr.name = "ngammasqr_num_" + param.name

            new_gamma_deno_sqr = (
                gamma_deno_sqr * (1 - 1 / taus_x_t) +
                T.sqr((mg - norm_grad) * (old_grad - mg)) / taus_x_t
            )

            new_gamma_deno_sqr.name = "ngammasqr_den_" + param.name

            gamma = T.sqrt(gamma_nume_sqr) / (T.sqrt(gamma_deno_sqr + eps) + \
                    self.gamma_reg)

            gamma.name = "gamma_" + param.name

            if self.gamma_clip and self.gamma_clip > -1:
                gamma = T.minimum(gamma, self.gamma_clip)

            momentum_step = gamma * mg
            corrected_grad_cand = (norm_grad + momentum_step) / (1 + gamma)

            #For starting the variance reduction.
            if self.start_var_reduction > -1:
                cond = T.le(self.start_var_reduction, step)
                corrected_grad = cond * corrected_grad_cand + (1 - cond) * norm_grad
            else:
                corrected_grad = norm_grad

            if self.use_adagrad:
                g = corrected_grad
                # Accumulate gradient
                new_sum_squared_grad = (
                    sum_square_grad + T.sqr(g)
                )
                rms_g_t = T.sqrt(new_sum_squared_grad)
                rms_g_t = T.maximum(rms_g_t, 1.0)

            #Use the gradients from the previous update
            #to compute the \nabla f(x_t) - \nabla f(x_{t-1})
            cur_curvature = norm_grad - old_plain_grad
            #cur_curvature = theano.printing.Print("Curvature: ")(cur_curvature)
            cur_curvature_sqr = T.sqr(cur_curvature)

            new_curvature_ave = (
                mean_curvature * (1 - 1 / taus_x_t) +
                (cur_curvature / taus_x_t)
            )
            new_curvature_ave.name = "ncurve_ave_" + param.name

            #Average average curvature
            nc_ave = new_curvature_ave / (1 - new_prod_taus)

            new_curvature_sqr_ave = (
                mean_curvature_sqr * (1 - 1 / taus_x_t) +
                (cur_curvature_sqr / taus_x_t)
            )
            new_curvature_sqr_ave.name = "ncurve_sqr_ave_" + param.name

            #Unbiased average squared curvature
            nc_sq_ave = new_curvature_sqr_ave / (1 - new_prod_taus)

            epsilon = 1e-7
            #lr_scalers.get(param, 1.) * learning_rate
            scaled_lr = sharedX(1.0)
            rms_dx_tm1 = T.sqrt(msdx + epsilon)

            rms_curve_t = T.sqrt(new_curvature_sqr_ave + epsilon)

            #This is where the update step is being defined
            delta_x_t = -scaled_lr * (rms_dx_tm1 / rms_curve_t - cov_num_t / (new_curvature_sqr_ave + epsilon))
            delta_x_t.name = "delta_x_t_" + param.name

            # This part seems to be necessary for only RNNs
            # For feedforward networks this does not seem to be important.
            if self.delta_clip:
                logger.info("Clipping will be applied on the adaptive step size.")
                delta_x_t = delta_x_t.clip(-self.delta_clip, self.delta_clip)
                if self.use_adagrad:
                    delta_x_t = delta_x_t * corrected_grad / rms_g_t
                else:
                    logger.info("Clipped adagrad is disabled.")
                    delta_x_t = delta_x_t * corrected_grad
            else:
                logger.info("Clipping will not be applied on the adaptive step size.")
                if self.use_adagrad:
                    delta_x_t = delta_x_t * corrected_grad / rms_g_t
                else:
                    logger.info("Clipped adagrad will not be used.")
                    delta_x_t = delta_x_t * corrected_grad

            new_taus_t = (1 - T.sqr(mdx) / (msdx + eps)) * taus_x_t + sharedX(1 + eps, "stabilized")

            #To compute the E[\Delta^2]_t
            new_mean_square_dx = (
                 msdx * (1 - 1 / taus_x_t) +
                 (T.sqr(delta_x_t) / taus_x_t)
             )

            #To compute the E[\Delta]_t
            new_mean_dx = (
                mdx * (1 - 1 / taus_x_t) +
                (delta_x_t / (taus_x_t))
            )

            #Perform the outlier detection:
            #This outlier detection is slightly different:
            new_taus_t = T.switch(T.or_(abs(norm_grad - mg) > (2 * T.sqrt(mgsq  - mg**2)),
                                        abs(cur_curvature - nc_ave) > (2 * T.sqrt(nc_sq_ave - nc_ave**2))),
                                        T.switch(new_taus_t > 2.5, sharedX(2.5), new_taus_t + sharedX(1.0) + eps), new_taus_t)

            #Apply the bound constraints on tau:
            new_taus_t = T.maximum(self.lower_bound_tau, new_taus_t)
            new_taus_t = T.minimum(self.upper_bound_tau, new_taus_t)

            new_cov_num_t = (
                cov_num_t * (1 - 1 / taus_x_t) +
                (delta_x_t * cur_curvature) * (1 / taus_x_t)
            )

            update_step = delta_x_t

            tot_norm_up += update_step.norm(2)
            # Apply updates
            updates[mean_square_grad] = new_mean_squared_grad
            updates[mean_square_dx] = new_mean_square_dx
            updates[mean_dx] = new_mean_dx
            updates[gnorm_sqr] = new_gnorm_sqr
            updates[gamma_nume_sqr] = new_gamma_nume_sqr
            updates[gamma_deno_sqr] = new_gamma_deno_sqr
            updates[taus_x_t] = new_taus_t
            updates[cov_num_t] = new_cov_num_t
            updates[mean_grad] = new_mean_grad
            updates[old_plain_grad] = norm_grad
            updates[mean_curvature] = new_curvature_ave
            updates[mean_curvature_sqr] = new_curvature_sqr_ave

            if self.perform_update:
                updates[param] = param + update_step

            updates[step] = step + 1
            updates[prod_taus] = new_prod_taus

            if self.use_adagrad:
                updates[sum_square_grad] = new_sum_squared_grad

            if self.use_corrected_grad:
                updates[old_grad] = corrected_grad

        f_update = theano.function([learning_rate], [tot_norm_up],
                                   updates=updates,
                                   on_unused_input='ignore')

        return f_grad_shared, f_update
