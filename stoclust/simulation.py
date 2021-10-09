"""
stoclust.simulation

Will contain functions for generating random walks
of various types. For now, only contains regulated
Markovian walks.

Functions
---------
markov_random_walk(probs,initial=None,group=None,regulator=None,halt=None,max_time=100,tol=1e-6):

    Given a set of transition probabilities, generates a random walk
    through the available nodes. This walk may be regulated by
    a regulator function or halting condition. See stoclust.regulators
    for further details.

"""

import numpy as _np
from stoclust import regulators as _regulators
from stoclust.Group import Group as _Group

def markov_random_walk(probs,initial=None,group=None,regulator=None,halt=None,max_time=100,tol=1e-6):
    """
    Given a set of transition probabilities, generates a random walk
    through the available nodes. The initial node, if not specified,
    is randomly selected. The method returns two items.
    The first is the report array R,
    dimensions Mx2, where the reports are indexed by the axis M.
    R[i,0] is the content of the report and R[i,1] is the time of the report.
    The second is the path P, an N-dimensional vector, where N is the number of steps
    in the random walk and P[j] is the node at time j.

    The walk may be regulated. This involves passing a regulator,
    which is a function that takes the simulation time, 
    the transition probabilities, the current node, 
    and an array of node data. At each time, 
    the regulator returns a Boolean indicating
    whether a report is to be made, the content of the report,
    and a new set of transition probabilities determined by
    the available information. The regulator also
    updates node_data in-place.

    Lastly, one can either specify a maximum length
    of time for the walk (using the keyword max_time) 
    or a more general halting condition (using the keyword halt).
    A halting condition takes the time, current node, and node data.
    If neither are specified, the maximum length will
    be set to 100 steps.

    Arguments
    ---------

    probs :         A square Markov matrix indicating the 
                    transition probabilities for the walk.

    initial :       The initial node. If group is not None, 
                    then the type of initial should be the 
                    type of the elements of group. Otherwise, 
                    initial should be the index of the initial node. 
                    If not specified, a random node will be chosen.

    group :         A Group whose elements label the indices of probs. 
                    If specified, inputs like initial and outputs 
                    like the path refer to nodes by their labels. 
                    If not specified, nodes will be referred to 
                    in inputs and outputs by their index.

    regulator :     The regulator determines how the probability matrix 
                    will be modified over the course of the walk, 
                    and what events will be noted in reports. 
                    See stoclust.regulators for more details. 
                    If not specified, a trivial regulator will be used 
                    which never modifies the transition matrix 
                    and returns no reports.

    halt :          The halt condition determines under what conditions 
                    the walk should stop. See the stoclust.regulators 
                    for more details. If not specified, a trivial halt 
                    condition to stop after a specified number of steps 
                    will be used; the number of steps can be changed 
                    using the max_time argument.

    max_time :      If a halt condition is not specified, then the walk 
                    is halted automatically after max_time steps. 
                    The default is set to 100.
    """
    if group is None:
        group = _Group(_np.arange(probs.shape[0]))

    if regulator is None:
        regulator = lambda t,ps,an,nd: (False,None,ps)

    if halt is None:
        halt = lambda t,an,nd: _regulators.halt_after_time(t,an,nd,max_time=max_time)

    if initial is None:
        initial_ind = _np.random.choice(_np.arange(probs.shape[0]))
    else:
        initial_ind = group.ind[initial]
    
    reports = []
    locations = []
    node_data = _np.zeros([probs.shape[0]])

    t = 0
    current = initial_ind
    will_report, report, new_probs = regulator(t,probs,current,node_data)
    if will_report:
            reports.append([report,t])
    locations.append(initial)
    t += 1

    while not(halt(t,current,node_data)):
        if _np.sum(new_probs[current,:])<tol:
            remaining = _np.where(_np.sum(new_probs,axis=0)>tol)[0]
            sequel = _np.random.choice(remaining)
        else:
            sequel = _np.random.choice(_np.arange(probs.shape[0]),p=new_probs[current,:])
        
        will_report, report, new_probs = regulator(t,probs,sequel,node_data)
        if will_report:
            reports.append([report,t])

        current = sequel
        locations.append(group.elements[current])
        t += 1
    
    return _np.array(reports), _np.array(locations)