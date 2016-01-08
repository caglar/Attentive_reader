from ..commons import EPS
import theano.tensor as TT

def logsumexp(x, axis=None):
    x_max = TT.max(x, axis=axis, keepdims=True)
    z = TT.log(TT.sum(TT.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    return z.sum(axis=axis)

def kl_divergence(p, p_hat):
    term1 = p * TT.log(TT.maximum(p, EPS))
    term2 = p * TT.log(TT.maximum(p_hat, EPS))
    term3 = (1 - p) * TT.log(TT.maximum(1 - p, EPS))
    term4 = (1 - p) * TT.log(TT.maximum(1 - p_hat, EPS))
    return term1 - term2 + term3 - term4


def running_ave(old_val, new_val, alpha=0.9):
    if old_val <= 0.:
        return new_val
    else:
        return alpha * old_val + (1. - alpha) * new_val
