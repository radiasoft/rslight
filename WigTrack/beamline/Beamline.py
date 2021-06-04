from ..propagators import Drift, Linear, ThinLens
from ..filters import SingleSlit

class Beamline:
    """Object for holding and manipulating beamline elements"""
    
    def __init__(self):
        
        self.elements = []
        
    
    def add_element(self, bl_elem):
        
        self.elements.append(bl_elem)
        
    
    def list_element_names(self):
        """List the names of the elements in order. Useful for making sure the sequence is right."""
        elem_names = []
        for elem in self.elements:
            elem_names.append(elem.name)
            
        return elem_names
    
    
    def propagate_beamline(self, wd):
        """Propagate a Wigner distribution through the beamline
        
        wd : Wigner distribution
            The distribution to be evolved through the beamline"""
        
        for elem in self.elements:
            elem.propagate(wd)