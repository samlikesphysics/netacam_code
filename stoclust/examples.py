"""
stoclust.ensemble

Contains functions for generating example data.

Functions
---------
gen_moon(rad=1.0,occ=0.5,num_samples=100):

    Generates random two-dimensional vectors, 
    arranged in a crescent moon shape.

gen_disk(rad1=1.0,rad2=2.0,num_samples=100):
    Generates random two-dimensional vectors, 
    arranged in an annulus.

"""

import numpy as _np

def gen_moon(rad=1.0,occ=0.5,num_samples=100):
    """
    Generates random two-dimensional vectors, arranged in a crescent moon shape.
    The shape is described by a circle of radius rad partially occluded by a circle of the same radius,
    with the degree of overlap (or occultation) given by occ.
    """
    sampled = 0
    samples = _np.zeros([num_samples,2])
    while sampled < num_samples:
        #print(samples)
        r = _np.sqrt(_np.random.rand())*rad
        theta = _np.random.rand()*2*_np.pi
        x = r*_np.cos(theta)
        y = r*_np.sin(theta)
        if _np.sqrt((x+occ)**2+y**2)>rad:
            samples[sampled,0] = x
            samples[sampled,1] = y
            sampled += 1
    return samples

def gen_disk(rad1=1.0,rad2=2.0,num_samples=100):
    """
    Generates random two-dimensional vectors, arranged in an annulus.
    The shape is described by a circle of radius rad2 with a middle circle of radius rad1 subtracted from the middle.
    """
    sampled = 0
    samples = _np.zeros([num_samples,2])
    while sampled < num_samples:
        #print(samples)
        r = _np.sqrt(_np.random.rand())*rad2
        theta = _np.random.rand()*2*_np.pi
        x = r*_np.cos(theta)
        y = r*_np.sin(theta)
        if r>rad1:
            samples[sampled,0] = x
            samples[sampled,1] = y
            sampled += 1
    return samples
