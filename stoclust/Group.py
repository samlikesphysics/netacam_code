import numpy as _np

class Group:
    """
    A cross between an array and a set, which contains
    no duplicate elements but allows indexing and can
    allows cross-referencing between a Group and its sub-Groups.

    A Group is defined by its initializing array containing elements,
    and optionally by an additional keyword parameter that
    points to a superset Group.

    Attributes that can be obtained are self.elements (the initializing array of elements),
    self.ind (a dictionary whose keys are elements and values are indices),
    and self.in_superset (an array whose indices correspond
    to self.elements, but whose entries give the corresponding indices
    for the same elements in superset.elements). A superset may be defined
    after instantiation using the method self.set_super.

    Aggregations act like arrays in that the elements
    may be called by indices. That is, for a group G, G[j]
    returns the same thing as G.elements[j]. Furthermore,
    G may be called as in iterator, which would yield the
    same result as calling G.elements as an iterator,
    and the boolean statement "i in G" yields the same
    result as "i in G.elements". len(G) is just the number of elements.
    Lastly, Groups may be added together, which results
    in the union of their elements, but the superset must be redefined
    afterwards.
    """
    def __init__(self,a,superset=None):
        if isinstance(a,list):
            self.elements = (_np.array(a))
        else:
            self.elements = (a)
        self.ind = {self.elements[i]:i for i in range(len(self.elements))}
        self.size = len(self.elements)
        self.in_superset = None
        if superset is not None:
            self.in_superset = _np.array([_np.where(superset.elements==s)[0][0] for s in self.elements])

    def set_super(self,superset):
        self.in_superset = _np.array([_np.where(superset.elements==s)[0][0] for s in self.elements])

    def __add__(self,other):
        IDs = _np.sort(_np.array(list(set(list(self.elements)+list(other.elements)))))
        new_group = Group(IDs)
        return new_group
    
    def __iter__(self):
        return iter(self.elements)

    def __repr__(self):
        return 'Group('+repr(list(self.elements))+')'

    def __getitem__(self,key):
        return self.elements[key]

    def __contains__(self,item):
        return item in self.elements

    def __len__(self):
        return self.size

    def __eq__(self,group):
        if set(list(self.elements))==set(list(group.elements)):
            return True
        else:
            return False