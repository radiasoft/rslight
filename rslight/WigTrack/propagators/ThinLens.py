from .Linear import Linear
import numpy as np
from copy import deepcopy

class ThinLens(Linear):
    """A thin lens element with a single focal length"""
    def __init__(self, f, name):
        """Initialize the element
        
        Args:
        
        f : float
            The focal length of the thin lens
             
        name : str
            The name of the element
        """
        
        self.f = f
        self.matrix = np.array([[1., 0.],
                                [-1./f, 1.]])
        
        self._elem_name = name
        self._elem_type = 'Matrix'

        
    @property
    def name(self):
        """Get the name of the element"""
        return self._elem_name
    
    @property
    def elem_type(self):
        """Get the name of the element"""
        return self._elem_type
    
    
    def propagate(self, wd):
        
        # move the shapes in phase space
        wd.coord_array = np.einsum('ij, kj -> ki', wd.coord_array)