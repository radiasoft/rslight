import numpy as np

from scipy.sparse import lil_matrix, csr_matrix

from numba import jit

# Class for transforming Wigner function through beamline
class WignerDistribution:
    """Class for storing and manipulating Wigner Distribution data"""

    def __init__(self, ns = [10, 10, 10, 10], Ls = [1., 1., 1., 1.]):

        
        self.dim = len(ns)
        self.n_cells = np.prod(ns)
        self.cell_dims = ns
        self.cell_lens = Ls
        self.cell_ds   = np.arrays(ns)/np.array(Ls)

        ## The create_grid function initializes all of the matrices needed for deposition
        self._create_grid()
        self.coord_array = self._get_grid_coordinates()
        
        # compute the base lambda matrix, which is a sparse matrix
        self._lambda_0 = self._compute_lambda_matrix(self.coord_array)
        
        self.kind = 'Wigner'


    def _create_grid(self):
        """Create the weights and coordinates"""
        
        
        self.wgt_array = np.zeros(self.n_cells)
        self._set_coordinates()
        
        # create the array of coordinates
        
        
        
        x_min = -self.L_x/2
        x_max = self.L_x/2
        y_min = -self.L_theta
        y_max = self.L_theta

        # Define the grid for the simulation
        self.x0 = np.linspace(x_min,x_max,self.n_x)
        self.y0 = np.linspace(y_min,y_max,self.n_theta)
        self.X0,self.Y0 = np.meshgrid(self.x0,self.y0)
        
        self.x0_p = self.X0.flatten()
        self.y0_p = self.Y0.flatten()

        self.x_grid_matrix, self.x_grid_matrix_t = np.meshgrid(self.x0_p,self.x0_p)
        self.y_grid_matrix, self.y_grid_matrix_t = np.meshgrid(self.y0_p,self.y0_p)

        self.lambda_matrix = self.tent_integral(np.zeros(len(self.x0_p)), np.zeros(len(self.x0_p)))
        self.lambda_matrix_inv = np.linalg.inv(self.lambda_matrix)
        print(self.lambda_matrix_inv)
        
    
    ##
    ## Deposition and related functions
    ##

    def deposit_to_grid(self):
    """Deposit the shifted grid points back to the original grid"""
    
        lambda_matrix = self._compute_lambda_matrix(self.coords)
        self.wgt_array = scipy.sparse.linalg.spsolve(self._lambda_0, lambda_matrix.dot(self.wgt_array))
        
    
    def _compute_lambda_matrix(self, coordinates):
    """Compute the convolution matrix for a set of coordinates back to the original grid
    
    Args:
    -----
    coordinates : (numpy array) an array of moved grid coordinates
    
    Returns:
    --------
    lambda_matrix : (sparse matrix) matrix of convolutions of each coordinate with the corresponding grid cells"""
    
    matrix = scipy.sparse.lil_matrix((self.n_cells, self.n_cells))
    
    for idx, coord in enumerate(coordinates):
        
        self._add_ptcl_wgt_lambda(idx, coord, matrix)
        
    return matrix


    @jit
    def _add_ptcl_wgt_lambda(self, idx, ptcl_pos, lambda_matrix):
        """Deposit the individual particle weights to the lambda matrix that defines the deposition
        
        Args:
        -----
        
        idx (int) : the index of the particle's original coordinate
        
        ptcl_pos (array) : the 2N-dimensional coordinates of the particle after moving
        
        lambda_matrix (sparse matrix) : the matrix being deposited to"""
        
        cell_idx = self._middle_index(ptcl_pos)
        
        frac_pos = (ptcl_pos/self.cell_dx) - cell_idx
        
        funcs = [lambda x: 1./6. * (-3. * x ** 3. - 6. * x**2. + 4.),
                 lambda x: -1./6. * (x - 2.) ** 3.,
                 lambda x: 1./6. * (x + 2.) ** 3,
                 lambda x: 1./6. * (3. * x** 3. - 6. * x** 2. + 4.)]
        
        # This can likely be done using recursion instead, which would be 
        # much more elegant and dimension-agnostic than nested loops -sdw 31 Aug 2021
        for i in range(0,4):
            
            idx0 = cell_idx[0] + i - 1
            wgt0 = funcs[i](frac_pos[0])
            
            for j in range(0,4):
                
                idx1 = cell_idx[1] + j
                wgt1 = funcs[j](frac_pos[1])
                
                for k in range(0,4):
                    
                    idx2 = cell_idx[2] + k
                    wgt2 = funcs[k](frac_pos[2])
                    
                    for l in range(0,4):
                        
                        idx3 = cell_idx[3] + l
                        wgt3 = funcs[l](frac_pos[3])
                        
                        stride_idx = 
                        
                        lambda_matrix[idx,]
        

    ##
    ## Grid Helper Functions
    ##
    
    @jit
    def _get_grid_coordinates(self):
        """Set the coordinates for each grid point."""
        
        coord_array = np.zeros((self.n_cells, self.dim))
        
        # compute the individual indices from the stride index
        for idx in range(self.n_cells):
            # this runs backwards, so will have to flip in the end
            for jdx, N_val in enumerate(np.flip(self.cell_dims)):
                coord_array[idx, jdx] = idx%N_val
                idx -= idx%N_val
                
        coord_array = np.flip(self.coord_array, axis=1)
        
        # scale and center indices to grid
        coord_array *= self.cell_ds
        coord_array -= self.cell_lens/2.
        
        return coord_array
        
        
    @jit   
    def _middle_index(self, coord_array):
        """Computes the linear stride index for the nearest neighbors of a coordinate array
        
        Args:
        -----
        
        coord_array : NumPy array of dimension (N_cells x Dim) that contains an array of coordinates
        
        Returns:
        --------
        
        linear_index : NumPy array of dimension (N_cells) that contains the linear stride index for the central
                       grid cell for each coordinate on the grid
        """
        
        assert coord_array.shape == (self.n_cells, self.dim)
        
        cell_ints = (coord_array/self.cell_ds).astype(int)
        middle_index = np.zeros(coord_array.shape[0], dtype='int')
        
        for idx, cell in enumerate(cell_ints):
            stride_loc = 0
            
            for jdx, val in enumerate(cell):
                stride_loc *= self.n_cells[jdx]
                stride_loc += val
            
            middle_index[idx] = stride_loc
            
        return middle_index
    
    
    @jit 
    def stride_to_index(self, stride_indexes):
        """Convert a stride index to a multi-dimensional index"""
        
        array_idx = np.array((len(stride_indexes), self.dim))

        for idx, val in enumerate(stride_indexes):
            
            for jdx, n_dir in np.flip(self.cell_dims):
                array_idx[idx, jdx] = val%n_dir
                val -= val%n_dir
                val = val // n_dir
                
            array_idx[idx] = np.flip(array_idx[idx])

        return array_idx
            
    @jit
    def index_to_stride(self, linear_indices):
        """Convert a linear index to a stride index"""
        
        stride_idx = np.array((len(linear_indices), 1))
        
        for idx, val in enumerate(linear_indices):
            
            for jdx, n_dir in self.cell_dims:
                stride_idx[idx,0] *= n_dir
                stride_idx[idx,0] += linear_indices[jdx]
                
        return stride_idx
                
            
            
######## Below lies the old stuff we are improving        
        

    def tent_integral(self,dx_vector,dy_vector):
        
        # Create two matricies that contain the motion of the particles
        dx_matrix, dx_matrix_t = np.meshgrid(dx_vector, dx_vector)
        dy_matrix, dy_matrix_t = np.meshgrid(dy_vector, dy_vector)

        # This is the kernel for the tent integral
        shift_x = self.x_grid_matrix_t + dx_matrix_t
        shift_y = self.y_grid_matrix_t + dy_matrix_t

        kernel_x = ( self.x_grid_matrix - shift_x ) / self.dx
        kernel_y = ( self.y_grid_matrix - shift_y ) / self.dtheta

        # We use the numpy piecewise function for the tent integral, first we define a list of lambda functions
        f_list = [  lambda x: 1./6. * (-3. * x ** 3. - 6. * x**2. + 4.),
                    lambda x: -1./6. * (x - 2.) ** 3.,
                    lambda x: 1./6. * (x + 2.) ** 3,
                    lambda x: 1./6. * (3. * x** 3. - 6. * x** 2. + 4.),
                    lambda x: 0.]

        # This computes the tent integral for both the x and y motion if there are no periodic boundary conditions
        out_y = np.piecewise(kernel_y, [(kernel_y >= -1.) * (kernel_y <= 0.),
             (kernel_y >= 1.) * (kernel_y <= 2.),
             (kernel_y >= -2.) * (kernel_y < -1.),
             (kernel_y > 0.) * (kernel_y < 1.),
             (kernel_y > 2.) + (kernel_y < -2.)], f_list)

        
        out_x = np.piecewise(kernel_x, [(kernel_x >= -1.) * (kernel_x <= 0.),
             (kernel_x >= 1.) * (kernel_x <= 2.),
             (kernel_x >= -2.) * (kernel_x < -1.),
             (kernel_x > 0.) * (kernel_x < 1.),
             (kernel_x > 2.) + (kernel_x < -2.)], f_list)

        return out_x * out_y * self.dx * self.dtheta



    def set_initial_weights(self, w0):
        """ Function to set the initial weights """

        self.wigner_grid = w0
    

    def update_weights(self,dx,dy):

        # Compute the matrix containing all the tent integrals for the shifted particle positons
        matrix_RHS = self.tent_integral(dx,dy)

        # Solve for the new weights based on the result of the tent integrals
        w1 = np.dot(np.dot(self.wigner_grid,matrix_RHS),self.lambda_matrix_inv)

        # Update the weights
        self.wigner_grid = w1
       
    
    def create_grid_SW(self):
        
        x_min = -self.L_x/2
        x_max = self.L_x/2
        th_min = -self.L_theta/2
        th_max = self.L_theta/2
        n_cells = self.n_x * self.n_theta
        
        # fill in the coordinates with linear strides
        coord_array = np.zeros((n_cells,2))
        coord_array[:,0] = np.repeat(np.linspace(x_min, x_max, self.n_x), self.n_theta)
        coord_array[:,1] = np.tile(np.linspace(th_min, th_max, self.n_theta), self.n_x)
        
        self.lambda_matrix = self.tent_integral(np.zeros(n_cells), np.zeros(n_cells))
        self.lambda_matrix_inv = np.linalg.inv(self.lambda_matrix)

        # create the weight grid
        self.wigner_grid = np.zeros(n_cells)
    
    
    def set_initial_weights_SW(self, w0):
        """Set the weights in Wigner grid"""
        
        
class ElectricField:
    """Class for storing and manipulating Wigner Distribution data"""

    def __init__(self, ns = [10, 10], Ls = [1., 1.]):

        
        self.dim = len(ns)
        self.n_cells = np.prod(ns)
        self.cell_dims = ns
        self.cell_lens = Ls
        self.cell_ds   = np.arrays(ns)/np.array(Ls)

        ## The create_grid function initializes all of the matrices needed for deposition
        self._create_grid()
        
        self.kind = 'Field'