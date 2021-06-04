import numpy as np

class SingleSlit:
    
    def __init__(self, center, width):
        
        
        self.center = center
        self.width  = width
        
    
    def slit_func(self, x):
                
        func = np.where(np.abs(x - self.center) < self.width, 1./(x[-1] - x[0]), 0.)
        
        return func
    
    def slit_filter(self, x, k):
        
        #x = x - self.center
        
        XX, KK = np.meshgrid(x, k)
        
        slit_filter = 2.*np.sin(KK*(-2.*np.abs(XX-self.center) + self.width))/(KK)
        slit_filter[np.where(np.abs(XX - self.center)>0.5*self.width)] = 0.
        
        return slit_filter
    
    
    def propagate(self, wd):
         
        wd.wigner_grid = signal.fftconvolve(slit_filter, wd.wigner_grid, axes=1, mode='same')*wd.dy