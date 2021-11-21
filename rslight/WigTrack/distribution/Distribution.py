import numpy as np

from scipy.sparse import lil_matrix, csr_matrix

from numba import jit

# Class for transforming Wigner function through beamline
class WignerDistribution:
    """Class for storing and manipulating Wigner Distribution data"""

    def __init__(self, ns = [10, 10, 10, 10], Ls = [1., 1., 1., 1.]):

        self.dim = len(ns)
        self.cell_ns = np.array(ns)
        self.n_cells = np.prod(ns)
        self.cell_dims = ns
        self.lens = np.array(Ls)
        self.cell_ds   = np.array(Ls)/np.array(self.cell_ns - 1.)

        # set the lambda_0 matrix used in the deposition. 
        # Just set the matrix once, as it is sparse and expensive to construct
        self.coord_array = self._get_grid_coordinates()
        print(self.coord_array)
        self._lambda_0 = self._compute_lambda_matrix(self.coord_array)
        
        self.kind = 'Wigner'
        
    
    ##
    ## Deposition and related functions
    ##

    def deposit_to_grid(self):
        """Deposit the shifted grid points back to the original grid"""
    
        lambda_matrix = self._compute_lambda_matrix(self.coords)
        self.wgt_array = scipy.sparse.linalg.spsolve(self._lambda_0, lambda_matrix.dot(self.wgt_array))
 

    @jit
    def _compute_lambda_matrix(self, coordinates):
        """Compute the convolution matrix for a set of coordinates back to the original grid

        Args:
        -----
        coordinates : (numpy array) an array of moved grid coordinates

        Returns:
        --------
        lambda_matrix : (sparse matrix) matrix of convolutions of each coordinate with the corresponding grid cells"""

        matrix = scipy.sparse.lil_matrix((self.n_cells, self.n_cells))

        if len(coord) == 2:
            lambda_comp = self._add_ptcl_wgt_lambda_2d
        else:
            lambda_comp = self._add_ptcl_wgt_lambda

        for idx, coord in enumerate(coordinates):
            lambda_comp(idx, coord, matrix)

        # benchmark csr versus csc for performance -sdw, Sep. 1 2021
        # matrix.tocsc()
        matrix.tocsr() 

        return matrix

    @jit
    def _add_ptcl_wgt_lambda_2d(self, idx, ptcl_pos, lambda_matrix):
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
                idx1 = cell_idx[1] + j - 1
                wgt1 = funcs[j](frac_pos[1])
                        
                stride_idx = self.index_to_stride([[idx0, idx1]])
                lambda_matrix[idx, stride_idx] += wgt0*wgt1


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
                idx1 = cell_idx[1] + j - 1
                wgt1 = funcs[j](frac_pos[1])
                
                for k in range(0,4):
                    idx2 = cell_idx[2] + k - 1
                    wgt2 = funcs[k](frac_pos[2])
                    
                    for l in range(0,4):
                        idx3 = cell_idx[3] + l -1
                        wgt3 = funcs[l](frac_pos[3])
                        
                        stride_idx = self.index_to_stride([[idx0, idx1, idx2, idx3]])
                        lambda_matrix[idx, stride_idx] += wgt0*wgt1*wgt2*wgt3
                                

    ##
    ## Grid Helper Functions
    ##
    
    @jit
    def _get_grid_coordinates(self):
        """Set the coordinates for each grid point."""
        
        print('entering get grid coordinates \n')
        
        coord_array = np.zeros((self.n_cells, self.dim))
        
        # compute the individual indices from the stride index
        for idx in range(self.n_cells):
            # this runs backwards, so will have to flip in the end
            for jdx, N_val in enumerate(np.flip(self.cell_dims)):
                tmp_idx = idx
                coord_array[idx, jdx] = (tmp_idx%N_val)
                tmp_idx -= tmp_idx%N_val
                                
        coord_array = np.flip(coord_array, axis=1)
        
        # scale and center indices to grid
        coord_array *= self.cell_ds
        coord_array -= self.lens/2.
        
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

        for idx, stride in enumerate(stride_indexes):
            
            for jdx, n_dir in np.flip(self.cell_dims):
                array_idx[idx, jdx] = stride%n_dir
                stride -= val%n_dir
                stride = stride // n_dir
                
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
                
            
            
    def set_distribution(self, grid):
        
        print(np.shape(grid))
        print(self.cell_ns)
        
        #if np.shape(grid) != self.cell_ns:
        #    except ValueError:
        #        print(
        #            'Shape of grid () does not match shape of distribution ()'.format(np.shape(grid), self.cell_ns)
        #        )
        #        raise
                
        self.wgt_array = np.flatten(grid)
        
    
    def get_distribution(self):
        
        return np.reshape(self.wgt_array, self.cell_ns)
        
        
class ElectricField:
    """Class for storing and manipulating Wigner Distribution data"""

    def __init__(self, ns = [10, 10], Ls = [1., 1.]):

        
        self.dim = len(ns)
        self.n_cells = np.prod(ns)
        self.cell_dims = ns
        self.lens = Ls
        self.cell_ds   = np.arrays(ns)/np.array(Ls)

        ## The create_grid function initializes all of the matrices needed for deposition
        self._create_grid()
        
        self.kind = 'Field'