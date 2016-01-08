"""
.. todo::

    WRITEME
"""
import logging
import warnings

import theano
import theano.tensor as T
from theano.compat.six.moves import input, zip as izip

import numpy
np = numpy

from theano.compat import six
from ..commons import EPS, global_rng
from functools import partial

WRAPPER_ASSIGNMENTS = ('__module__', '__name__')
WRAPPER_CONCATENATIONS = ('__doc__',)
WRAPPER_UPDATES = ('__dict__',)

logger = logging.getLogger(__name__)

#Groundhog related imports
import numpy
import random
import string
import copy as pycopy

import theano
import theano.tensor as TT
import inspect
import re

import os


def ensure_dir_exists(directory):
    """
    If the dir does not exist recreate it
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def overrides(method):
    """Decorator to indicate that the decorated method overrides a method
    in superclass.
    The decorator code is executed while loading class. Using this method should
    have minimal runtime performance implications.

    This is based on my idea about how to do this and fwc:s highly improved
    algorithm for the implementation
    fwc:s algorithm : http://stackoverflow.com/a/14631397/308189
    answer : http://stackoverflow.com/a/8313042/308189

    How to use:
    from overrides import overrides

    class SuperClass(object):

        def method(self):
            return 2

    class SubClass(SuperClass):

        @overrides
        def method(self):
            return 1

    :raises  AssertionError if no match in super classes for the method name
    :return  method with possibly added (if the method doesn't have one)
    docstring from super class
    @NOTE: This is based on pip overrides package.
    """
    stack = inspect.stack()
    base_classes = [s.strip() for s in re.search(r'class.+\((.+)\)\s*:', \
            stack[2][4][0]).group(1).split(',')]
    if not base_classes:
        raise ValueError("overrides decorator: unable to determine base class"
                         "for method %s" % method.__name__)
    # replace each class name in base_classes with the actual class type
    derived_class_locals = stack[2][0].f_locals
    for i, base_class in enumerate(base_classes):
        if '.' not in base_class:
            base_classes[i] = derived_class_locals[base_class]
        else:
            components = base_class.split('.')
            # obj is either a module or a class
            obj = derived_class_locals[components[0]]
            for c in components[1:]:
                assert(inspect.ismodule(obj) or inspect.isclass(obj))
                obj = getattr(obj, c)
            base_classes[i] = obj
    for super_class in base_classes:
        if hasattr(super_class, method.__name__):
            if not method.__doc__:
                method.__doc__ = getattr(super_class, method.__name__).__doc__
            return method
    raise AssertionError('No super class method found for "%s"' % method.__name__)


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = TT.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = TT.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


def kl_divergence(p, p_hat):
    term1 = p * T.log(p)
    term2 = p * T.log(p_hat)
    term3 = (1-p) * T.log(1 - p + EPS)
    term4 = (1-p) * T.log(1 - p_hat + EPS)
    return term1 - term2 + term3 - term4


def sparsity_penalty(h, sparsity_level=0.05, sparse_reg=1e-4):
    if h.ndim == 2:
        sparsity_level = T.extra_ops.repeat(sparsity_level, h.shape[1])
    else:
        sparsity_level = T.extra_ops.repeat(sparsity_level, h.shape[0])

    sparsity_penalty = 0
    avg_act = h.mean(axis=0)
    kl_div = self.kl_divergence(sparsity_level, avg_act)
    sparsity_penalty = sparse_reg * kl_div.sum()

    # Implement KL divergence here.
    return sparsity_penalty


def get_key_byname_from_dict(dict_, name):
    keys = dict_.keys()
    keyval = None
    for key in keys:
        if key.name == name:
            keyval = key
            break

    return keyval


def print_time(secs):
    if secs < 120.:
        return '%6.3f sec' % secs
    elif secs <= 60 * 60:
        return '%6.3f min' % (secs / 60.)
    else:
        return '%6.3f h  ' % (secs / 3600.)


def print_mem(context=None):
    if theano.sandbox.cuda.cuda_enabled:
        rvals = theano.sandbox.cuda.cuda_ndarray.cuda_ndarray.mem_info()
        # Avaliable memory in Mb
        available = float(rvals[0]) / 1024. / 1024.
        # Total memory in Mb
        total = float(rvals[1]) / 1024. / 1024.
        if context == None:
            print ('Used %.3f Mb Free  %.3f Mb, total %.3f Mb' %
                   (total - available, available, total))
        else:
            info = str(context)
            print (('GPU status : Used %.3f Mb Free %.3f Mb,'
                    'total %.3f Mb [context %s]') %
                    (total - available, available, total, info))


def safe_grad(cost, params, known_grads=None):
    from collections import OrderedDict
    grad_list = T.grad(cost, params, known_grads=known_grads, add_names=True)
    grads = OrderedDict({param: grad for param, grad in safe_izip(params, grad_list)})
    return grads


def const(value):
    return TT.constant(numpy.asarray(value, dtype=theano.config.floatX))


def as_floatX(variable):
    """
       This code is taken from pylearn2:
       Casts a given variable into dtype config.floatX
       numpy ndarrays will remain numpy ndarrays
       python floats will become 0-D ndarrays
       all other types will be treated as theano tensors
    """

    if isinstance(variable, float):
        return numpy.cast[theano.config.floatX](variable)

    if isinstance(variable, numpy.ndarray):
        return numpy.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)


def copy(x):
    new_x = pycopy.copy(x)
    new_x.params = [x for x in new_x.params]
    new_x.params_grad_scale      = [x for x in new_x.params_grad_scale    ]
    new_x.noise_params           = [x for x in new_x.noise_params         ]
    new_x.noise_params_shape_fn  = [x for x in new_x.noise_params_shape_fn]
    new_x.updates                = [x for x in new_x.updates              ]
    new_x.additional_gradients   = [x for x in new_x.additional_gradients ]
    new_x.inputs                 = [x for x in new_x.inputs               ]
    new_x.schedules              = [x for x in new_x.schedules            ]
    new_x.properties             = [x for x in new_x.properties           ]
    return new_x


def softmax(x):
    if x.ndim == 2:
        e = TT.exp(x)
        return e / TT.sum(e, axis=1).dimshuffle(0, 'x')
    else:
        e = TT.exp(x)
        return e/ TT.sum(e)


def id_generator(size=5, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for i in xrange(size))


def constant_shape(shape):
    return lambda *args, **kwargs : shape


def binVec2Int(binVec):
    add = lambda x,y: x+y
    return reduce(add,
                  [int(x) * 2 ** y
                   for x, y in zip(
                       list(binVec),range(len(binVec) - 1, -1,
                                                       -1))])


def Int2binVec(val, nbits=10):
    strVal = '{0:b}'.format(val)
    value = numpy.zeros((nbits,), dtype=theano.config.floatX)
    if theano.config.floatX == 'float32':
        value[:len(strVal)] = [numpy.float32(x) for x in strVal[::-1]]
    else:
        value[:len(strVal)] = [numpy.float64(x) for x in strVal[::-1]]
    return value


def dot(inp, matrix):
    """
    Decide the right type of dot product depending on the input
    arguments
    """
    if 'int' in inp.dtype and inp.ndim >= 2:
        return matrix[inp.flatten()]
    elif 'int' in inp.dtype:
        return matrix[inp]
    elif 'float' in inp.dtype and inp.ndim == 3:
        shape0 = inp.shape[0]
        shape1 = inp.shape[1]
        shape2 = inp.shape[2]
        return TT.dot(inp.reshape((shape0*shape1, shape2)), matrix)
    elif 'float' in inp.dtype and inp.ndim == 2:
        return TT.dot(inp, matrix)
    else:
        return TT.dot(inp, matrix)


def dbg_hook(hook, x):
    if not isinstance(x, TT.TensorVariable):
        x.out = theano.printing.Print(global_fn=hook)(x.out)
        return x
    else:
        return theano.printing.Print(global_fn=hook)(x)


def make_name(variable, anon="anonymous_variable"):
    """
    If variable has a name, returns that name. Otherwise, returns anon.

    Parameters
    ----------
    variable : tensor_like
        WRITEME
    anon : str, optional
        WRITEME

    Returns
    -------
    WRITEME
    """

    if hasattr(variable, 'name') and variable.name is not None:
        return variable.name

    return anon


def sharedX(value, name=None, borrow=False, dtype=None, broadcastable=None):
    """
    Transform value into a shared variable of type floatX

    Parameters
    ----------
    value : WRITEME
    name : WRITEME
    borrow : WRITEME
    dtype : str, optional
        data type. Default value is theano.config.floatX

    Returns
    -------
    WRITEME
    """

    if dtype is None:
        dtype = theano.config.floatX
    return theano.shared(numpy.cast[dtype](value),
                         name=name,
                         borrow=borrow,
                         broadcastable=broadcastable)


def as_floatX(variable):
    """
    Casts a given variable into dtype `config.floatX`. Numpy ndarrays will
    remain numpy ndarrays, python floats will become 0-D ndarrays and
    all other types will be treated as theano tensors

    Parameters
    ----------
    variable : WRITEME

    Returns
    -------
    WRITEME
    """

    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)

    return theano.tensor.cast(variable, theano.config.floatX)


def constantX(value):
    """
    Returns a constant of value `value` with floatX dtype

    Parameters
    ----------
    variable : WRITEME

    Returns
    -------
    WRITEME
    """
    return theano.tensor.constant(np.asarray(value,
                                             dtype=theano.config.floatX))


def subdict(d, keys):
    """
    Create a subdictionary of d with the keys in keys

    Parameters
    ----------
    d : WRITEME
    keys : WRITEME

    Returns
    -------
    WRITEME
    """
    result = {}
    for key in keys:
        if key in d:
            result[key] = d[key]
    return result


def safe_update(dict_to, dict_from):
    """
    Like dict_to.update(dict_from), except don't overwrite any keys.

    Parameters
    ----------
    dict_to : WRITEME
    dict_from : WRITEME

    Returns
    -------
    WRITEME
    """
    for key, val in six.iteritems(dict_from):
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to


class CallbackOp(theano.gof.Op):
    """
    A Theano Op that implements the identity transform but also does an
    arbitrary (user-specified) side effect.

    Parameters
    ----------
    callback : WRITEME
    """
    view_map = {0: [0]}

    def __init__(self, callback):
        self.callback = callback

    def make_node(self, xin):
        """
        .. todo::

            WRITEME
        """
        xout = xin.type.make_variable()
        return theano.gof.Apply(op=self, inputs=[xin], outputs=[xout])

    def perform(self, node, inputs, output_storage):
        """
        .. todo::

            WRITEME
        """
        xin, = inputs
        xout, = output_storage
        xout[0] = xin
        self.callback(xin)

    def grad(self, inputs, output_gradients):
        """
        .. todo::

            WRITEME
        """
        return output_gradients

    def R_op(self, inputs, eval_points):
        """
        .. todo::

            WRITEME
        """
        return [x for x in eval_points]

    def __eq__(self, other):
        """
        .. todo::

            WRITEME
        """
        return type(self) == type(other) and self.callback == other.callback

    def hash(self):
        """
        .. todo::

            WRITEME
        """
        return hash(self.callback)

    def __hash__(self):
        """
        .. todo::

            WRITEME
        """
        return self.hash()


def get_dataless_dataset(model):
    """
    Loads the dataset that model was trained on, without loading data.
    This is useful if you just need the dataset's metadata, like for
    formatting views of the model's weights.

    Parameters
    ----------
    model : Model

    Returns
    -------
    dataset : Dataset
        The data-less dataset as described above.
    """

    global yaml_parse
    global control

    if yaml_parse is None:
        from pylearn2.config import yaml_parse

    if control is None:
        from pylearn2.datasets import control

    control.push_load_data(False)
    try:
        rval = yaml_parse.load(model.dataset_yaml_src)
    finally:
        control.pop_load_data()
    return rval


def safe_zip(*args):
    """Like zip, but ensures arguments are of same length"""
    base = len(args[0])
    for i, arg in enumerate(args[1:]):
        if len(arg) != base:
            raise ValueError("Argument 0 has length %d but argument %d has "
                             "length %d" % (base, i+1, len(arg)))
    return zip(*args)


def safe_izip(*args):
    """Like izip, but ensures arguments are of same length"""
    assert all([len(arg) == len(args[0]) for arg in args])
    return izip(*args)


def gpu_mem_free():
    """
    Memory free on the GPU

    Returns
    -------
    megs_free : float
        Number of megabytes of memory free on the GPU used by Theano
    """
    global cuda
    if cuda is None:
        from theano.sandbox import cuda
    return cuda.mem_info()[0]/1024./1024


class _ElemwiseNoGradient(theano.tensor.Elemwise):
    """
    A Theano Op that applies an elementwise transformation and reports
    having no gradient.
    """

    def connection_pattern(self, node):
        """
        Report being disconnected to all inputs in order to have no gradient
        at all.

        Parameters
        ----------
        node : WRITEME
        """
        return [[False]]

    def grad(self, inputs, output_gradients):
        """
        Report being disconnected to all inputs in order to have no gradient
        at all.

        Parameters
        ----------
        inputs : WRITEME
        output_gradients : WRITEME
        """
        return [theano.gradient.DisconnectedType()()]

# Call this on a theano variable to make a copy of that variable
# No gradient passes through the copying operation
# This is equivalent to making my_copy = var.copy() and passing
# my_copy in as part of consider_constant to tensor.grad
# However, this version doesn't require as much long range
# communication between parts of the code
block_gradient = _ElemwiseNoGradient(theano.scalar.identity)

def is_block_gradient(op):
    """
    Parameters
    ----------
    op : object

    Returns
    -------
    is_block_gradient : bool
        True if op is a gradient-blocking op, False otherwise
    """

    return isinstance(op, _ElemwiseNoGradient)


def safe_union(a, b):
    """
    Does the logic of a union operation without the non-deterministic ordering
    of python sets.

    Parameters
    ----------
    a : list
    b : list

    Returns
    -------
    c : list
        A list containing one copy of each element that appears in at
        least one of `a` or `b`.
    """
    if not isinstance(a, list):
        raise TypeError("Expected first argument to be a list, but got " +
                        str(type(a)))
    assert isinstance(b, list)
    c = []
    for x in a + b:
        if x not in c:
            c.append(x)
    return c

# This was moved to theano, but I include a link to avoid breaking
# old imports
from theano.printing import hex_digest as _hex_digest
def hex_digest(*args, **kwargs):
    warnings.warn("hex_digest has been moved into Theano. "
            "pylearn2.utils.hex_digest will be removed on or after "
            "2014-08-26")

def function(*args, **kwargs):
    """
    A wrapper around theano.function that disables the on_unused_input error.
    Almost no part of pylearn2 can assume that an unused input is an error, so
    the default from theano is inappropriate for this project.
    """
    return theano.function(*args, on_unused_input='ignore', **kwargs)


def grad(*args, **kwargs):
    """
    A wrapper around theano.gradient.grad that disable the disconnected_inputs
    error. Almost no part of pylearn2 can assume that a disconnected input
    is an error.
    """
    return theano.gradient.grad(*args, disconnected_inputs='ignore', **kwargs)


# Groups of Python types that are often used together in `isinstance`
if six.PY3:
    py_integer_types = (int, np.integer)
    py_number_types = (int, float, complex, np.number)
else:
    py_integer_types = (int, long, np.integer)  # noqa
    py_number_types = (int, long, float, complex, np.number)  # noqa

py_float_types = (float, np.floating)
py_complex_types = (complex, np.complex)


def get_choice(choice_to_explanation):
    """
    .. todo::

        WRITEME

    Parameters
    ----------
    choice_to_explanation : dict
        Dictionary mapping possible user responses to strings describing
        what that response will cause the script to do

    Returns
    -------
    WRITEME
    """
    d = choice_to_explanation

    for key in d:
        logger.info('\t{0}: {1}'.format(key, d[key]))
    prompt = '/'.join(d.keys())+'? '

    first = True
    choice = ''
    while first or choice not in d.keys():
        if not first:
            warnings.warn('unrecognized choice')
        first = False
        choice = input(prompt)
    return choice


def float32_floatX(f):
    """
    This function changes floatX to float32 for the call to f.
    Useful in GPU tests.

    Parameters
    ----------
    f : WRITEME

    Returns
    -------
    WRITEME
    """
    def new_f(*args, **kwargs):
        """
        .. todo::

            WRITEME
        """
        old_floatX = theano.config.floatX
        theano.config.floatX = 'float32'
        try:
            f(*args, **kwargs)
        finally:
            theano.config.floatX = old_floatX

    # If we don't do that, tests function won't be run.
    new_f.func_name = f.func_name
    return new_f

