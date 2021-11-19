from rslight.WigTrack.distribution import Distribution

class Simulation:
    
    def __init__(self):
        
        self._lattice = []
        
    
    def create_simulation(self, )
    
    def create_distribution(self, params_dict, init_dist):
        
        ns = params_dict['n_cells']
        ls = params_dict['Ls']
        
        self._distribution = Distribution(np.array(ns), np.array(ls))
        
        self._distribution.set_initial_distribution(init_dist)
        
    