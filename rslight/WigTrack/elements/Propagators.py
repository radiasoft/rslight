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
        new_coords = deepcopy(wd.coord_grid)
        new_coords[:,:,1] -= wd.coord_grid[:,:,0]/self.f
        
        wd.update_weights(new_coords)