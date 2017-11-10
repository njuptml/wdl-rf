import autograd.numpy as np
import autograd.numpy.random as npr

import os
import csv
import itertools as it

import sys, signal, pickle
from contextlib import contextmanager
from time import time
from functools import partial
from collections import OrderedDict

def collect_test_losses(num_folds):
    # Run this after CV results are in. e.g:
    # python -c "from deepmolecule.util import collect_test_losses; collect_test_losses(10)"
    results = {}
    for net_type in ['conv', 'morgan']:
        results[net_type] = []
        for expt_ix in range(num_folds):
            fname = "Final_test_loss_{0}_{1}.pkl.save".format(expt_ix, net_type)
            try:
                with open(fname) as f:
                    results[net_type].append(pickle.load(f))
            except IOError:
                print "Couldn't find file {0}".format(fname)

    print "Results are:"
    print results
    print "Means:"
    print {k : np.mean(v) for k, v in results.iteritems()}
    print "Std errors:"
    print {k : np.std(v) / np.sqrt(len(v) - 1) for k, v in results.iteritems()}

def record_loss(loss, expt_ix, net_type):
    fname = "Final_test_loss_{0}_{1}.pkl.save".format(expt_ix, net_type)
    with open(fname, 'w') as f:
        pickle.dump(float(loss), f)

def N_fold_split(N_folds, fold_ix, N_data):
    fold_ix = fold_ix % N_folds
    fold_size = N_data / N_folds
    test_fold_start = fold_size * fold_ix
    test_fold_end   = fold_size * (fold_ix + 1)
    test_ixs  = range(test_fold_start, test_fold_end)
    train_ixs = range(0, test_fold_start) + range(test_fold_end, N_data)
    return train_ixs, test_ixs

def rmse(X, Y):
    return np.sqrt(np.mean((X - Y)**2))

def R2(x,y):
    mean=np.mean(y)
    t1=np.sum((y-x)**2)
    t2=np.sum((y-mean)**2)
    return 1-t1/t2
    
def Rs2(x,y):
    xmean=np.mean(x)
    ymean=np.mean(y)
    s1=np.sum((x-xmean)*(y-ymean))**2
    s2=np.sum((x-xmean)**2)*np.sum((y-ymean)**2)
    return s1/s2

def slicedict(d, ixs):
    return {k : v[ixs] for k, v in d.iteritems()}

class memoize(object):
    def __init__(self, func):
        self.func = func
        self.cache = {}

    def __call__(self, *args):
        if args in self.cache:
            return self.cache[args]
        else:
            result = self.func(*args)
            self.cache[args] = result
            return result

    def __get__(self, obj, objtype):
        return partial(self.__call__, obj)

def normalize_array(A):
    mean, std = np.mean(A), np.std(A)
    A_normed = (A - mean) / std
    def restore_function(X):
        return X * std + mean

    return A_normed, restore_function

@contextmanager
def tictoc():
    print "--- Start clock ---"
    t1 = time()
    yield
    dt = time() - t1
    print "--- Stop clock: %s seconds elapsed ---" % dt

class WeightsParser(object):
    """A kind of dictionary of weights shapes,
       which can pick out named subsets from a long vector.
       Does not actually store any weights itself."""
    def __init__(self):
        self.idxs_and_shapes = OrderedDict()
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        """Takes in a vector and returns the subset indexed by name."""
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)

    def set(self, vect, name, value):
        """Takes in a vector and returns the subset indexed by name."""
        idxs, _ = self.idxs_and_shapes[name]
        vect[idxs] = np.ravel(value)

    def __len__(self):
        return self.N

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return map(lambda s: x == s, allowable_set)

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)

def get_ith_minibatch_ixs(i, num_datapoints, batch_size):
    num_minibatches = num_datapoints / batch_size + ((num_datapoints % batch_size) > 0)
    i = i % num_minibatches
    start = i * batch_size
    stop = start + batch_size
    return slice(start, stop)

def build_batched_grad(grad, batch_size, inputs, targets):
    """Grad has signature(weights, inputs, targets)."""
    def batched_grad(weights, i):
        cur_idxs = get_ith_minibatch_ixs(i, len(targets), batch_size)
        return grad(weights, inputs[cur_idxs], targets[cur_idxs])
    return batched_grad

def dropout(weights, fraction, random_state):
    """Randomly sets fraction of weights to zero, and increases the rest
        such that the expected activation is the same."""
    mask = random_state.rand(len(weights)) > fraction
    return weights * mask / (1 - fraction)

def add_dropout(grad, dropout_fraction, seed=0):
    """Actually, isn't this dropconnect?"""
    assert(dropout_fraction < 1.0)
    def dropout_grad(weights, i):
        mask = npr.RandomState(seed * 10**6 + i).rand(len(weights)) > dropout_fraction
        masked_weights = weights * mask / (1 - dropout_fraction)
        return grad(masked_weights, i)
    return dropout_grad

def catch_errors(run_fun, catch_fun):
    def signal_term_handler(signal, frame):
        catch_fun()
        sys.exit(0)
    signal.signal(signal.SIGTERM, signal_term_handler)
    try:
        result = run_fun()
    except:
        catch_fun()
        raise

    return result
    
def read_csv(filename, nrows, input_name, target_name):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in it.islice(reader, nrows):
            data[0].append(row[input_name])
            data[1].append(float(row[target_name]))
    return map(np.array, data)

def load_data(filename, sizes, input_name, target_name):
    slices = []
    start = 0
    for size in sizes:
        stop = start + size
        slices.append(slice(start, stop))
        start = stop
    return load_data_slices_nolist(filename, slices, input_name, target_name)

def load_data1(filename, input_name):
    data=[]
    with open(filename) as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            data.append(row[input_name])
    return np.array(data)
    
def load_data2(filename, input_name, target_name):
    data = ([], [])
    with open(filename) as file:
        reader = csv.DictReader(file)
        for row in reader:
            data[0].append(float(row[input_name]))
            data[1].append(float(row[target_name]))
    return data
    
    
def load_data_slices_nolist(filename, slices, input_name, target_name):
    stops = [s.stop for s in slices]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv(filename, max(stops), input_name, target_name)
    return [(data[0][s], data[1][s]) for s in slices]


def list_concat(lists):
    return list(it.chain(*lists))
    
def load_data_slices(filename, slice_lists, input_name, target_name):
    stops = [s.stop for s in list_concat(slice_lists)]
    if not all(stops):
        raise Exception("Slices can't be open-ended")

    data = read_csv(filename, max(stops), input_name, target_name)

    return [(np.concatenate([data[0][s] for s in slices], axis=0),
             np.concatenate([data[1][s] for s in slices], axis=0))
            for slices in slice_lists]

def get_output_file(rel_path):
    return os.path.join(output_dir(), rel_path)

def get_data_file(rel_path):
    return os.path.join(data_dir(), rel_path)

def output_dir():
    return os.path.expanduser(safe_get("OUTPUT_DIR"))

def data_dir():
    return os.path.expanduser(safe_get("DATA_DIR"))

def safe_get(varname):
    if varname in os.environ:
        return os.environ[varname]
    else:
        raise Exception("%s environment variable not set" % varname)