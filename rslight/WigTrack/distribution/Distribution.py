import numpy as np

# Class for transforming Wigner function through beamline
class WignerDistribution:
    """Class for storing and manipulating Wigner Distribution data"""

    def __init__(self, n_x = 10, n_theta = 10, L_x = 1, L_theta = 1):

        self.n_x = n_x
        self.n_theta = n_theta
        self.L_x = L_x
        self.L_theta = L_theta
        self.wigner_grid = np.zeros([self.n_x, self.n_theta])
        self.dx = L_x/n_x
        self.dtheta = L_theta/n_theta

        ## The create_grid function initializes all of the matrices needed for deposition
        self.create_grid()


    def create_grid(self):
        
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
