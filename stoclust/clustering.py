"""
stoclust.clustering

Contains functions providing basic clustering techniques
motivated by stochastic analysis.

Functions
---------
split_by_gaps(vec,num_gaps = 1,group = None)

    Aggregates the indices of a vector 
    based on gaps between index values.
    The number of gaps is specified by num_gaps,
    and the largest num_gaps gaps in the sorted array
    are used to cluster values.

split_by_quantiles(vec,quantiles=0.95,group = None)

    Like split_by_vals, but cuts the vector at specific quantiles
    rather than rigid values. Assumes right-continuity of the 
    cumulative distribution function.

split_by_vals(vec,cuts=0,group = None,tol=0)

    Aggregates the indices of a vector based on specified
    values at which to cut the sorted array. Assumes the
    right-continuity of the cumulative distribution function.

meyer_wessell(st_mat,min_times_same = 5,vector_clustering = None,group = None)

    Given a column-stochastic matrix describing the strength
    of the relationship between pairs of items,
    determines an aggregation of the items using the dynamical
    approach of Meyer and Wessell.

shi_malik(st_mat,eig_thresh=0.95,cut=0,group=None)

    Given a stochastic matrix describing the strength
    of the relationship between pairs of items,
    determines an aggregation of the items using
    the spectral approach of Shi and Malik.

fushing_mcassey(st_mat,max_visits=5,time_quantile_cutoff=0.95,group=None)

    Given a square stochastic matrix describing the strength
    of the relationship between pairs of items,
    determines an aggregation of the items using
    the regulated random walk approach of Fushing and McAssey.

hier_from_blocks(block_mats,scales=None,group=None)

    Given a parameterized ensemble of block matrices, 
    each more coarse-grained than the last,
    constructs a corresponding Hierarchy object.

"""

import numpy as _np
import scipy.linalg as _la
import stoclust.utils as _utils
from stoclust.Group import Group as _Group
from stoclust.Aggregation import Aggregation as _Aggregation
from stoclust.Hierarchy import Hierarchy as _Hierarchy
import stoclust.simulation as _simulation
import stoclust.regulators as _regulators

def split_by_gaps(vec,num_gaps = 1,group = None):
    """
    Aggregates the indices of a vector based on gaps between index values.
    The number of gaps is specified by num_gaps,
    and the largest num_gaps gaps in the sorted array
    are used to cluster values.

    Arguments
    ---------
    vec :       A one-dimensional array of values.

    Keyword Arguments
    -----------------
    num_gaps :  The number of gaps to use to break vec into clusters.

    group :     The group which labels the indices of vec, and which will be the item set of the returned Aggregation.

    Output
    ------
    Aggregation of the indices of vec
    """
    if group is None:
        group = _Group(_np.arange(len(vec)))

    sort_inds = _np.argsort(vec)

    gap_locs = _np.flip(
        _np.argsort(
            _np.diff(vec[sort_inds])
        )
    )[0:num_gaps]

    ordered_gaps = _np.sort(gap_locs)

    agg_dict = {
        k+1:sort_inds[ordered_gaps[k]+1:ordered_gaps[k+1]+1] 
        for k in range(0,num_gaps-1)
    }
    agg_dict.update({
        0:sort_inds[:ordered_gaps[0]+1],
        num_gaps:sort_inds[ordered_gaps[-1]+1:]
    })
    return _Aggregation(group,_Group(_np.arange(num_gaps+1)),agg_dict)

def split_by_quantiles(vec,quantiles=0.95,group = None):
    """
    Like split_by_vals, but cuts the vector at specific quantiles
    rather than rigid values. Assumes right-continuity of the 
    cumulative distribution function.

    Arguments
    ---------
    vec :       A one-dimensional array of values.

    Keyword Arguments
    -----------------
    quantiles : A single value or list/array of quantiles which will be used to divide the vector components.

    group :     The group which labels the indices of vec, and which will be the item set of the returned Aggregation.

    Output
    ------
    Aggregation of the indices of vec
    """
    num = len(vec)
    if group is None:
        group = _Group(_np.arange(len(vec)))

    if not(isinstance(quantiles,_np.ndarray)):
        if isinstance(quantiles,list):
            quantiles = _np.array(quantiles)
        else:
            quantiles = _np.array([quantiles])

    cdf = _np.sum(vec[:,None] <= vec[None,:],axis=0)/len(vec)
    where = quantiles[:,None]<= cdf[None,:]
    cuts = _np.amin(
        vec[None,:]*where,
        initial=_np.amax(vec),
        where=where,
        axis=1
    )

    return split_by_vals(vec,cuts=cuts,group=group)

def split_by_vals(vec,cuts=0,group = None,tol=0):
    """
    Aggregates the indices of a vector based on specified
    values at which to cut the sorted array. Assumes the
    right-continuity of the cumulative distribution function.

    Arguments
    ---------
    vec :   A one-dimensional array of values.

    Keyword Arguments
    -----------------
    cuts :  A single value or list/array of values which will be used to divide the vector components.

    group : The group which labels the indices of vec, and which will be the item set of the returned Aggregation.

    Output
    ------
    Aggregation of the indices of vec
    """
    if group is None:
        group = _Group(_np.arange(len(vec)))

    if not(isinstance(cuts,_np.ndarray)):
        if isinstance(cuts,list):
            cuts = _np.array(cuts)
        else:
            cuts = _np.array([cuts])
    
    cuts = cuts[
        _np.where(
            _np.logical_and(
                cuts>=_np.amin(vec),
                cuts<_np.amax(vec)
            )
        )[0]
    ]

    if len(cuts)==0:
        return _Aggregation(
            group,
            _Group(_np.array([0])),
            {0:_np.arange(len(vec))}
        )
    else:
        agg_dict = {
            k+1:_np.where(
                _np.logical_and(
                    vec > cuts[k],
                    vec <= cuts[k+1]
                )
            )[0]
            for k in range(0,len(cuts)-1)
        }
        agg_dict.update({
            0:_np.where(vec <= cuts[0])[0],
            len(cuts):_np.where(vec > cuts[len(cuts)-1])[0]
        })

        return _Aggregation(
            group,
            _Group(_np.arange(len(cuts)+1)),
            agg_dict
        )

def meyer_wessell(st_mat,min_times_same = 5,vector_clustering = None,group = None):
    """
    Given a column-stochastic matrix describing the strength
    of the relationship between pairs of items,
    determines an aggregation of the items using the dynamical
    approach of Meyer and Wessell. The algorithm
    is inherently random, though fairly stable, and so may
    be used as a one-shot measure but will be more reliable
    in an ensemble.

    A column-stochastic matrix T will, by the Perron-Frobenius theorem,
    have a uniform vector u = (1,...,1) as a fixed point, that is:

    T u = u
    
    This fixed point is unique as long as 
    the stochastic matrix is not reducible
    into disconnected components. If it is almost reducible
    (that is, if there are strongly connected communities with
    weak connections between them), the vector T^t u will achieve
    uniformity among the connected components before achieving global
    uniformity. 

    The Meyer-Wessell approach relies on applying the
    column-stochastic matrix T to a random initial vector x and
    detecting communities by identifying clusters of components which
    achieve uniformity long before global uniformity is reached.
    This is done by iteratively applying T to x and, at each iteration,
    performing some kind of vector clustering on T^t x. When
    the resulting Aggregation ceases to change ove a long
    enough number of iterations, it is returned as the final Aggregation.

    Arguments
    ---------
    st_mat :            A square bistochastic matrix.

    Keyword Arguments
    -----------------
    min_times_same :    The number of iterations after which, if the clustering has not changed, the algorithm halts.

    vector_clustering : The particular method of vector clustering which should be used in the algorithm.
                        Should receive a vector as the sole i_nput and return an Aggregation.

    group :             The group which labels the indices of st_mat, and which will be the item set of the returned Aggregation.

    Output
    ------
    Aggregation of the indices of st_mat
    """
    method = vector_clustering
    if group is None:
        group = _Group(_np.arange(st_mat.shape[0]))

    if method is None:
        eigs,vecs = _la.eig(st_mat)
        eig_agg = split_by_gaps(eigs)
        k = len(eig_agg[1])
        if k>1:
            method = lambda v:split_by_gaps(
                v,num_gaps = k-1,group = group
            )
        else:
            method = lambda v:_Aggregation(
                group,
                _Group(_np.array([0])),
                {0:_np.arange(len(group))}
            )

    x = _np.random.rand(st_mat.shape[0])
    times_same = 1
    new_agg = method(x)
    by_cluster = method(x).by_cluster()
    while times_same < min_times_same:
        x = st_mat@x
        new_agg = method(x)
        new_by_cluster = new_agg.by_cluster()
        if _np.all(new_by_cluster == by_cluster):
            times_same += 1
        else:
            times_same = 1
        by_cluster = new_by_cluster

    return new_agg
    
def shi_malik(st_mat,eig_thresh=0.95,cut=0,group=None):
    """
    Given a stochastic matrix describing the strength
    of the relationship between pairs of items,
    determines an aggregation of the items using
    the spectral approach of Shi and Malik.

    A column-stochastic matrix T will always have a leading
    eigenvalue of 1 and a leading uniform right-eigenvector, 
    u=(1,...,1), which is a fixed point of the map:

    T u = u

    If T has no disconnected components then u is the
    unique fixed point (up to a constant scaling) 
    and the sub-leading eigenvalue
    is strictly less than one; otherwise, the eigenvalue
    1 is degenerate. In the first case, if the sub-leading
    eigenvalue is close to 1, then the sub-leading
    right-eigenvector y may be used to partition the indices into
    two slowly-decaying communities.

    The Shi-Malik algorithm is recursive, taking
    the sub-leading eigenvector of T (as long as the
    corresponding eigenvalue is above a threshold),
    using it to bipartition the indices, and then
    repeating these steps on the partitions with a reweighted
    matrix. This implementation cuts the vector y by value,
    by default into components y>0 and y<=0, because of the
    orthogonality relationship

    <y>_pi = y . pi = 0

    which indicates that the mean value of y
    under the stationary distribution pi 
    (left-eigenvector of T)
    must always be zero, making this a value of significance.

    The algorithm halts when no community has a sub-leading
    eigenvector above the threshold, and the final partitioning
    is returned as an Aggregation.

    Arguments
    ---------
    st_mat :        A square stochastic matrix describing a Markov dynamic.

    Keyword Arguments
    -----------------
    eig_thresh :    The smallest value the subleading eigenvalue may have to continue the recursion.

    cut :           The value used to "cut" the subleading eigenvector into two clusters.

    group :         The group which labels the indices of st_mat, and which will be the item set of the returned Aggregation.

    Output
    ------
    Aggregation of the indices of st_mat
    """
    if group is None:
        group = _Group(_np.arange(st_mat.shape[0]))

    num_items = group.size
    clusts = _Aggregation(
        group,
        _Group(_np.array([0])),
        {0:_np.arange(len(group))}
    )
    change = True
    while change:
        new_clusts = []
        change = False
        for k,c in clusts:
            if len(c)>1:
                T = _utils.stoch(st_mat[
                    _np.ix_(c.in_superset,c.in_superset)
                ])
                eigs,evecs = _la.eig(T)
                einds = _np.flip(_np.argsort(_np.abs(eigs)))
                if eigs[einds[1]]>eig_thresh:
                    y = _np.real(evecs[:,einds[1]])
                    ind_agg = split_by_vals(y/_np.sum(y),group=c,cuts=cut)
                    if ind_agg.clusters.size>1:
                        new_clusts.append(c.in_superset[ind_agg[0].in_superset])
                        new_clusts.append(c.in_superset[ind_agg[1].in_superset])
                    else:
                        ind_agg = split_by_gaps(y,group=c)
                        new_clusts.append(c.in_superset[ind_agg[0].in_superset])
                        new_clusts.append(c.in_superset[ind_agg[1].in_superset])
                    change = True
                else:
                    new_clusts.append(c.in_superset)
            else:
                new_clusts.append(c.in_superset)

        new_agg = {j:new_clusts[j] for j in range(len(new_clusts))}
        clusts = _Aggregation(
            group,
            _Group(_np.arange(len(new_clusts))),
            new_agg
        )
    return clusts
        
def fushing_mcassey(st_mat,max_visits=5,time_quantile_cutoff=0.95,group=None):
    """
    Given a square stochastic matrix describing the strength
    of the relationship between pairs of items,
    determines an aggregation of the items using
    the regulated random walk approach of Fushing and McAssey.
    The algorithm is inherently random
    and highly unstable as a single-shot approach,
    but may be used in an ensemble to determine a 
    useful similarity matrix.

    Suppose st_mat is given by the Markov matrix T.
    A regulated random walk is taken using T as the initial
    transition probabilities, and modifying these probabilities
    to remove from circulation any node which has been visited
    at least max_visits times (this prevents the walk from
    being stuck in a cluster for too long). The time between removals
    is recorded; the highest values (determined by time_quantile_cutoff)
    determine the number of clusters (it is interpreted that a sudden long
    removal time after many short removal times indicates 
    one has left a highly-explored cluster and entered an unexplored one).

    A node which was removed and for which >50% of its visits
    prior to removal were in particular time-interval is placed in the cluster
    associated with that time interval; all other nodes remain unclustered.

    This algorithm will not return useful results after a single run,
    but if an ensemble of runs is collected it may be used to
    derive a similarity matrix, based on how often two nodes are in
    a cluster together over the many runs.

    Arguments
    ---------
    st_mat :                A square stochastic matrix describing a Markov dynamic.

    Keyword Arguments
    -----------------
    max_visits :            The maximum number of visits to a node before it is removed in the regulated random walk.

    time_quantile_cutoff :  The quantile of the length of time between node removals, which is used to determine the number of clusters.

    group :                 The group which labels the indices of st_mat, and which will be the item set of the returned Aggregation.

    Output
    ------
    Aggregation of the indices of st_mat
    """
    if group is None:
        group = _Group(_np.arange(st_mat.shape[0]))

    reg = lambda t,ps,an,nd: _regulators.node_removal_regulator(
        t,ps,an,nd,
        max_visits=max_visits
    )

    hlt = lambda t,an,nd: _regulators.halt_when_explored(
        t,an,nd
    )

    reports, path = _simulation.markov_random_walk(
        st_mat,
        regulator=reg,
        halt=hlt,
        group=group
    )

    if len(reports)==0:
        return _Aggregation(
            group,
            group,
            {
                j:_np.array([j]) 
                for j in _np.arange(group.size)
            }
        )
    else:
        times = _np.concatenate([
            _np.array([reports[0,1]]),
            _np.diff(reports[:,1])
        ])

        clust_times = reports[
            split_by_quantiles(
                -times,
                quantiles=_np.array([1-time_quantile_cutoff])
            )[0].in_superset,1
        ]

        if clust_times[0]>0:
            clust_times = _np.concatenate([_np.array([0]),clust_times])
        clusters = {
            j+group.size:[] 
            for j in range(len(clust_times))
        }
        
        for k in group:
            if k in _np.unique(reports[:,0]):
                block_counts = _np.add.reduceat((path==k),clust_times)
                props = block_counts/_np.sum(block_counts)
                if _np.any(props>0.5):
                    clusters[
                        _np.where(props>0.5)[0][0]+group.size
                    ].append(group.ind[k])
                else:
                    clusters[group.ind[k]] = [group.ind[k]]
            else:
                clusters[group.ind[k]] = [group.ind[k]]

        cluster_names = _np.array(
            [k for k in clusters.keys() if len(clusters[k])>0]
        )

        agg_dict = {
            j:_np.array(clusters[cluster_names[j]]) 
            for j in range(len(cluster_names))
        }

        return _Aggregation(
            group,
            _Group(cluster_names),
            agg_dict
        )

def hier_from_blocks(block_mats,scales=None,group=None):
    """
    Given a parameterized ensemble of block matrices, each more coarse-grained than the last,
    constructs a corresponding Hierarchy object.

    Arguments
    ---------
    block_mats :    A three-dimensional array. The first dimension is the ensemble dimension, and the remaining two dimensions are equal.

    Keyword Arguments
    -----------------
    scales :        A one-dimensional monotonically increasing array, giving a scale parameter for each ensemble

    group :         The group which labels the indices of block_mats.shape[1], and which will be the item set of the returned Aggregation.

    Output
    ------
    Hierarchy of the indices of block_mats.shape[1]
    """
    if group is None:
        group = _Group(_np.arange(block_mats.shape[1]))
    if scales is None:
        scales = _np.arange(block_mats.shape[0])

    current_agg = _Aggregation(
        group,
        _Group(_np.arange(group.size)),
        {j:_np.array([j]) for j in range(group.size)}
    )
    new_block_mats = block_mats.copy()
    cluster_children = {}
    proper_clusters = 0
    for j in range(block_mats.shape[0]):
        st_mat = _utils.stoch(new_block_mats[j])
        agg = shi_malik(
            st_mat,
            eig_thresh=0.99,
            group=current_agg.clusters,
            cut=1e-6
        )
        new_agg_dict = {}
        new_agg_names = []
        num_clusters = 0
        for k,c in agg:
            if c.size>1:
                cluster_children[proper_clusters+group.size] = (
                    scales[j],
                    current_agg.clusters.elements[c.in_superset]
                )
                new_agg_dict[num_clusters] = c.in_superset
                new_agg_names.append(proper_clusters+group.size)
                proper_clusters += 1
                num_clusters += 1
            else:
                new_agg_dict[num_clusters] = c.in_superset
                new_agg_names.append(c[0])
                num_clusters += 1
        new_agg = _Aggregation(
            current_agg.clusters,
            _Group(_np.array(new_agg_names)),
            new_agg_dict
        )

        blocks = [
            list(new_agg[new_agg.clusters[k]].in_superset) 
            for k in range(new_agg.clusters.size)
        ]
        new_block_mats = (
            _utils.block_sum(
                _utils.block_sum(
                    new_block_mats,
                    blocks,
                    axis=1
                ),
                blocks,
                axis=2
            ) > 0
        ).astype(float)
        current_agg = new_agg

    labels = _Group(_np.array(
        list(group.elements)+
        list(cluster_children.keys())
    ))

    return _Hierarchy(
        group,
        labels,
        cluster_children
    )