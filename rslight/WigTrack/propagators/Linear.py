import numpy as np
from numpy import einsum

class Linear:
    """Class for generic linear transformation of the Wigner distribution"""
    
    def __init__(self, matrix, name):
        """Initialize the element
        
        Args:
        
        matrix : numpy array
            The matrix that represents this generic element
             
        name : str
            The name of the element
        """
        
        assert type(name) == str, "Linear element name arg not a str"
        self._elem_name = name
        self.matrix = matrix
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
        """Propagate a Wigner distribution through this element
        
        Args:
        -----
        
        wd : WignerDistribution class
            The Wigner distribution to be propagated
        """
        
        new_coords = einsum('ij, abj -> abi', self.matrix, wd.coord_grid)
        
        wd.update_weights(new_coords)
        