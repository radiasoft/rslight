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
        
        self.name      = name
        self.elem_type = 'Matrix'

        
    @property
    def name(self):
        """Get the name of the element"""
        return self.name
    
    @property
    def elem_type(self):
        """Get the name of the element"""
        return self.elem_type
    
    
    def propagate(self, wd):
        
        # move the shapes in phase space
        new_coords = deepycopy(wd.coord_grid)
        new_coords[:,:,0] += wd.coord_grid[:,:,1]*L
        
        wd.update_weights(new_coords)