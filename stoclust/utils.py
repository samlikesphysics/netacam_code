"""
stoclust.utils

Contains miscellaneous useful functions.

Functions
---------
stoch(weights,axis=-1):

    Reweights each row along a given axis 
    such that sums along that axis are one.

sinkhorn_knopp(mat,num_iter=100,rescalings=False):

    Given a matrix of weights, generates a bistochastic matrix 
    using the Sinkhorn-Knopp algorithm.

block_sum(a, blocks, axis=0):

    Given an array and a list of index blocks
    along a given axis, delivers a new array whose indices
    correspond to blocks and whose
    entries are sums over the old values.
    
"""

import numpy as _np
from functools import reduce as _reduce

def stoch(weights,axis=-1):
    """
    Reweights each row along a given axis such that sums along that axis are one.
    """
    if isinstance(axis,tuple):
        axes = list(axis)
    elif axis==-1:
        axes = [len(weights.shape)-1]
    else:
        axes = [axis]
    shape = _np.array(weights.shape)
    all_axes = list(range(len(weights.shape)))
    other_axes = list(set(all_axes) - set(axes))
    reorg = axes + other_axes
    new_weights = _np.moveaxis(weights,reorg,all_axes)
    new_targets = list(range(len(axes)))
    sums = _np.sum(new_weights,axis=tuple(new_targets))
    prop_sums = sums + (sums==0).astype(int)
    sums_full = _np.outer(_np.ones([_np.prod(shape[(axes)])]),prop_sums).reshape(list(shape[(axes)])
                                                            +list(shape[(other_axes)]))
    return _np.moveaxis(new_weights/sums_full,all_axes,reorg)

def sinkhorn_knopp(mat,num_iter=100,rescalings=False):
    """
    Given a matrix of weights, generates a bistochastic matrix using the Sinkhorn-Knopp algorithm.

    A bistochastic matrix is one where the sum of rows
    is always equal to one, and the sum of columns is always
    equal to one.
    
    For any matrix M, there is a unique pair of
    diagonal matrices D_L and D_R such that
    D_L^(-1) M D_R^(-1) is bistochastic. The
    Sinkhorn-Knopp algorithm determines these matrices
    iteratively. This function will return the resulting
    bistochastic matrix and, optionally, the diagonal weights
    of the rescaling matrices.
    """
    dR = _np.ones([mat.shape[1]])
    dL = _np.ones([mat.shape[0]])
    for j in range(num_iter):
        dL = _np.sum(mat@_np.diag(dR),axis=1)**(-1)
        dR = _np.sum(_np.diag(dL)@mat,axis=0)**(-1)
    if rescalings:
        return dL, _np.diag(dL)@mat@_np.diag(dR), dR
    else:
        return _np.diag(dL)@mat@_np.diag(dR)

def block_sum(a, blocks, axis=0):
    """
    Given an array and a list of index blocks
    along a given axis, delivers a new array whose indices
    correspond to blocks and whose
    entries are sums over the old values.
    """
    if axis==0:
        move_a = a.copy()
    else:
        move_a = _np.moveaxis(a,[0,axis],[axis,0])
    order = _reduce(lambda x,y:x+y, blocks,[])
    order_a = move_a[order]
    lengths = _np.array([len(b) for b in blocks])
    slices = _np.concatenate([_np.array([0]),_np.cumsum(lengths)[:-1]])
    red_a = _np.add.reduceat(order_a,slices,axis=0)
    if axis==0:
        final_a = red_a.copy()
    else:
        final_a = _np.moveaxis(red_a,[0,axis],[axis,0])
    return final_a