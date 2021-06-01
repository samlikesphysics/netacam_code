import numpy as np

'''
Some useful scripts for working with Lorenz curves.
'''

def order(p,q):
    '''
    Returns the sorted indices (greatest to least)
    for the ratio p_i/q_i.
    '''
    return np.flip(np.argsort(p/q))

def rank(p,q):
    '''
    Assigns each index i a rank
    according to the value of p_i/q_i,
    ranked greatest to least.
    '''
    js = order(p,q)
    rank = np.empty_like(js)
    rank[js] = np.arange(len(js))
    return rank

def in_hull(p, hull):
    '''
    Determines if some set of points is in the convex
    hull of another set of points.
    '''
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)
    return hull.find_simplex(p)>=0

def is_majorized(p1,q1,p2,q2):
    '''
    Determines if two pairs of distributions
    are majorized.
    '''
    x1,y1 = lorenz_curve(p1,q1)
    x2,y2 = lorenz_curve(p2,q2)

    hull = np.array([x1,y1]).T
    p = np.array([x2,y2]).T
    return np.all(in_hull(p,hull))

def dismajorization(p1,q1,p2,q2):
    '''
    Calculates the dismajorization
    of two pairs of distributions.
    '''
    x1,y1 = lorenz_curve(p1,q1)
    x2,y2 = lorenz_curve(p2,q2)
    js = order(p2,q2)

    hull = np.array([x1,y1]).T
    p = np.array([x2[1:],y2[1:]]).T
    return 1-(in_hull(p,hull)*p2[js]).sum()

def lorenz_curve(p,q):
    '''
    Returns the x and y coordinates of the points of the Lorenz curve.
    '''
    js = order(p,q)
    ys = np.concatenate([np.array([0]),np.cumsum(p[js])])
    xs = np.concatenate([np.array([0]),np.cumsum(q[js])])
    return xs,ys