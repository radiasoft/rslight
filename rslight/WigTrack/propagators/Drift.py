from .Linear import Linear
import numpy as np
from copy import deepcopy

class Drift(Linear):
    """A drift element with a length L"""
    def __init__(self, L, name):
        """Initialize the element
        
        Args:
        
        L : float
            The length of the drift
             
        name : str
            The name of the element
        """        
        self.f = L
        self.matrix = np.array([[1., L],
                                [0., 1.]])
        
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
    
    
    def propagate_wigner(self, wd):
        
        # move the shapes in phase space
        new_coords = deepcopy(wd.coord_grid)
        new_coords[:,:,0] += wd.coord_grid[:,:,1]*L
        
        wd.update_weights(new_coords)
        
        
    def propagate_field(self, fd):
        
        return 0