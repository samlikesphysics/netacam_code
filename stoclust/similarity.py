import numpy as _np
from stoclust import ensemble as _ensemble

def from_gram(g):
    norms = _np.diag(g)
    return g/_np.sqrt((norms[None,:]*norms[:,None]))

def from_random_clustering(weights,**kwargs):
    parisi_ensemble = _ensemble.random_clustering(weights,**kwargs)
    return _np.average(parisi_ensemble,axis=0)