"""
stoclust.regulators

Contains some pre-defined regulators and halting
conditions for use in the markov_random_walk method.

A regulator is any function which receives the following arguments...

    time :      int                 The timestep of the random walk.


    probs :     (N,N) np.ndarray    The original transition probabilities of the walk.
                                    N is the number of nodes.

    at_node :   int                 The index of the current node.

    node_data : np.ndarray          An array in which "node data" may be stored,
                                    to make the random walk memoryful.

...and returns the following outputs...

    is_report : bool            Whether a report is to be logged.

    report :    object          The report, if is_report == True, 
                                or NoneType if is_report == False.

    new_probs : np.ndarray      The modified transition probabilities.

...and, optionally, which modifies the node_data array IN-PLACE
in order to update the memory of the walk.

A halting condition is similar but also simpler. It is any function
which receives the following arguments...

    time :      int             The timestep of the random walk.

    at_node :   int             The index of the current node.

    node_data : np.ndarray      An array in which "node data" may be stored,
                                to make the random walk memoryful.

...and returns a single boolean output, which if True will result
in the halting of the random walk.

Functions
---------
node_removal_regulator(time,probs,at_node,node_data,max_visits=5):

    Modifies transition probabilities by making it impossible to transition
    to any node which has been visited already at least max_visits times.
    Uses node_data to track the number of previous visits.
    When a node is removed, reports the ID of the removed node;
    otherwise, no report is made.

halt_when_explored(time,at_node,node_data):

    Halts when all nodes have been visited at least once.
    Assumes that node_data is being used to count visits;
    does not work otherwise. Therefore, be careful to use
    with an appropriate regulator.

halt_after_time(time,at_node,node_data,max_time=100):

    Halts after a fixed duration of time.

"""

import numpy as _np
from stoclust import utils as _utils

def node_removal_regulator(time,probs,at_node,node_data,max_visits=5):
    """
    Modifies transition probabilities by making it impossible to transition
    to any node which has been visited already at least max_visits times.
    Uses node_data to track the number of previous visits.
    When a node is removed, reports the ID of the removed node;
    otherwise, no report is made.
    """
    old_ind = _np.where(node_data>=max_visits)[0]
    node_data[at_node] += 1
    new_ind = _np.where(node_data>=max_visits)[0]
    new_probs = _utils.stoch(probs*((node_data<max_visits).astype(int)[None,:]))
    updated = _np.where(_np.logical_not(_np.isin(new_ind,old_ind)))[0]
    
    if len(updated>0):
        return True, new_ind[updated[0]], new_probs
    else:
        return False, None, new_probs

def halt_when_explored(time,at_node,node_data):
    """
    Halts when all nodes have been visited at least once.
    Assumes that node_data is being used to count visits;
    does not work otherwise. Therefore, be careful to use
    with an appropriate regulator.
    """
    return _np.all(node_data>0)

def halt_after_time(time,at_node,node_data,max_time=100):
    """
    Halts after a fixed duration of time.
    """
    return time>=max_time