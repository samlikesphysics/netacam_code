"""
stoclust.distance

Contains functions providing calculation
of basic distance metrics from raw data.

Functions
---------
euclid(vecs):

    Given a set of vectors, determines the Euclidean distance between them.

from_gram(g):

    Given a Gram matrix g, computes the corresponding
    distance of the basis vectors.

kl_div(probs):
    Given a column-stochastic matrix (that is,
    _np.sum(probs,axis=1)=1), generates a square
    matrix of dimension probs.shape[0],
    whose i,j element is the KL divergence
    of probs[i] over probs[j].

kl_div_cross(probs):
    Given a square column-stochastic matrix (that is,
    _np.sum(probs,axis=1)=1), generates a square
    matrix of dimension probs.shape[0],
    whose i,j component is the cross KL divergence
    of probs[i] over probs[j] (see this function's
    documentation for further details).

"""

import numpy as _np
import scipy.special as _sp
from stoclust import gram as _gram

def kl_div(probs):
    """
    Given a column-stochastic matrix (that is,
    _np.sum(probs,axis=1)=1), generates a square
    matrix of dimension probs.shape[0],
    whose i,j element is the KL divergence
    of probs[i] over probs[j].
    """
    dist = _sp.kl_div(probs[:,None,:],probs[None,:,:])
    return _np.sum(dist,axis=-1)

def euclid(vecs):
    """
    Given a set of vectors, determines the Euclidean distance between them.
    """
    g = _gram.from_vecs(vecs)
    return from_gram(g)

def from_gram(g):
    """
    Given a Gram matrix g, computes the corresponding
    distance of the basis vectors.
    """
    norms = _np.diag(g)
    return norms[None,:]+norms[:,None]-2*g

def kl_div_cross(probs):
    """
    Given a square column-stochastic matrix (that is,
    _np.sum(probs,axis=1)=1), generates a square
    matrix of dimension probs.shape[0],
    whose i,j element is given by the formula

    d[i,j] = sum_(k != i,j) probs[i,k] log(probs[i,k]/probs[j,k])
             + probs[i,j] log(probs[i,j]/probs[j,i])
             + probs[i,i] log(probs[i,i]/probs[j,j])
    """
    dist = _sp.kl_div(probs[:,None,:],probs[None,:,:])
    offdiags = _sp.kl_div(probs,probs.T)
    diags = _sp.kl_div(_np.diag(probs)[:,None],_np.diag(probs)[None,:])
    dist[_np.arange(probs.shape[0]),:,_np.arange(probs.shape[0])] = 0
    dist[:,_np.arange(probs.shape[0]),_np.arange(probs.shape[0])] = 0
    return _np.sum(dist,axis=-1) + diags + offdiags