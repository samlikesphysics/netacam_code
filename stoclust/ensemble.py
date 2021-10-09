"""
stoclust.ensemble

Contains functions for generating ensembles from data
and calculating clusters over ensembles.

Functions
---------
random_clustering(mat,clustering_method,ensemble_size=100,show_progress=False):

    Given a matrix and a function 
    describing a random clustering method, 
    returns an ensemble of block matrices.

given_ensemble(mats,clustering_method,show_progress=False):

    Given an ensemble of matrices and a clustering method, 
    returns the result of clustering each trial 
    as an ensemble of block matrices.

smooth_block(block_mats,window=7,cutoff=0.5):

    Given an ordered ensemble of block matrices, 
    uses a smoothing method to ensure hierarchical consistency.

from_noise(vecs,noise_map,ensemble_size=100,show_progress=False):

    Given an array of vectors and a function which takes each vector 
    to a randomly-generated ensemble, creates the ensemble 
    of randomly-generated instances for each given vector.

"""

import numpy as _np
from tqdm import tqdm as _tqdm
import scipy.linalg as _la
from functools import reduce as _reduce
from math import floor as _floor
from math import ceil as _ceil

def random_clustering(mat,clustering_method,ensemble_size=100,show_progress=False):
    """
    Given a matrix and a function describing a random clustering method, returns an ensemble of block matrices.

    Arguments
    ---------
    mat :               A square array, of whatever format is required by the clustering_method.

    clustering_method : Any function which takes a square matrix and returns an Aggregation; ideally one which uses random methods.

    Keyword Arguments
    -----------------
    ensemble size :     The number of ensembles to run.

    show_progress :     Boolean; whether or not to display a tqdm progress bar.

    Output
    ------
    block_mats :        A three-dimensional array whose first dimension indexes the ensemble trials, and whose remaining two indices have the shape of mat.
    """
    if show_progress:
        ensemble_iter = _tqdm(range(ensemble_size))
    else:
        ensemble_iter = range(ensemble_size)
    return _np.stack([clustering_method(mat).block_mat() for k in ensemble_iter])

def given_ensemble(mats,clustering_method,show_progress=False):
    """
    Given an ensemble of matrices and a clustering method, returns the result of clustering each trial as an ensemble of block matrices.

    Arguments
    ---------
    mat :               A three dimensional array, the first dimension of which is the ensemble index and the remaining two are square.

    clustering_method : Any function which takes a square matrix and returns an Aggregation.

    Keyword Arguments
    -----------------
    show_progress :     Boolean; whether or not to display a tqdm progress bar.

    Output
    ------
    block_mats :        A three-dimensional array the same shape as mat.
    """
    if show_progress:
        ensemble_iter = _tqdm(range(mats.shape[0]))
    else:
        ensemble_iter = range(mats.shape[0])
    return _np.stack([clustering_method(mats[j]).block_mat() for j in ensemble_iter])

def smooth_block(block_mats,window=7,cutoff=0.5):
    """
    Given an ordered ensemble of block matrices, uses a smoothing method to ensure hierarchical consistency.

    Arguments
    ---------
    block_mats :    A three dimensional array, the first dimension of which is the ensemble index and the remaining two are square.
    Keyword Arguments
    -----------------
    window :        The window to be used in the smoothing technique.

    cutoff :        The cutoff for whether a smoothed value should be set to 0 or 1.

    Output
    ------
    new_block_mats : A three-dimensional array the same shape as block_mats.
    """
    height = block_mats.shape[0]
    width = block_mats.shape[1]
    indices = _reduce(lambda x,y:x+y,[[_np.arange(max(0,j-_floor(window/2)),
                                                min(height,j+_ceil(window/2)))[0],
                                        _np.arange(max(0,j-2),min(height,j+3))[-1]] for j in range(height)],[])
    sums = _np.add.reduceat(block_mats,indices,axis=0)[0::2]
    nums = _np.add.reduceat(_np.ones([height]),indices,axis=0)[0::2]
    bmats_avg = sums/_np.outer(nums,_np.ones([width,width])).reshape([height,width,width])
    upper = (bmats_avg>cutoff).astype(float)
    new_bmats = _np.stack([(_la.expm(upper[j])>0).astype(float) for j in range(height)])
    result = _np.zeros([height,width,width])
    result[-1] = new_bmats[-1]
    for j in range(height-1):
        result[height-j-2] = new_bmats[height-j-2]*result[height-j-1]
    return result

def from_noise(vecs,noise_map,ensemble_size=100,show_progress=False):
    """
    Given an array of vectors and a function which takes each vector to a randomly-generated ensemble, creates the ensemble of randomly-generated instances for each given vector.

    Arguments
    ---------
    vecs :              A two-dimensional array, the first dimension of which indexes the vectors, and the remaining dimension is the dimension of the vectors.

    noise_map :         A function which takes a vector and a size parameter (for the ensemble size), and generates an ensemble of randomly generated vectors.
    
    Keyword Arguments
    -----------------
    ensemble size :     The number of ensembles to generate.

    show_progress :     Boolean; whether or not to display a tqdm progress bar.

    Output
    ------
    vec_ens :           A three-dimensional array whose first dimension indexes the ensemble trials, and whose remaining two indices have the shape of vecs.
    """
    if show_progress:
        vecs_iter = _tqdm(range(vecs.shape[0]))
    else:
        vecs_iter = range(vecs.shape[0])
    return _np.moveaxis(_np.stack([noise_map(vecs[k],ensemble_size) for k in vecs_iter]),
                       [0,1,2],[1,0,2])