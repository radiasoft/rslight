import numpy as np
from numpy import pi as PI
from numpy.fft import fft, ifft, fftshift, fftfreq, fftshift
from scipy.integrate import simps, trapz

def WignerTransformFunc(func, x_vals, k_vals): 
    """Computes the 1D Wigner transform of a function on a given grid
    
    Args:
    -----
    
    func : function
        The function to take the Wigner transform of. Takes only an array as an argument
    
    x_vals : array
        The points in x where the function should be evaluated
        
    k_vals : array
        The points in k-space where the function should be evaluated
        
    Returns:
    --------
    
    wig_func : numpy array
        The Wigner function on a grid, W(x, k)
        
    k_vals : numpy array
        The k-vector values for the Wigner function grid, related to the x-values input by the FFT frequencies"""
        
    # get the values of the function in the shape of
    # the wigner function output
    f = func(x_vals)
    
    return _WignerTransform(f, x_vals, k_vals)

def WignerTransformFuncFFT(func, x_vals, k_vals): 
    """Computes the 1D Wigner transform of a function on a given grid
    
    Args:
    -----
    
    func : function
        The function to take the Wigner transform of. Takes only an array as an argument
    
    x_vals : array
        The points in x where the function should be evaluated
        
    k_vals : array
        The points in k-space where the function should be evaluated
        
    Returns:
    --------
    
    wig_func : numpy array
        The Wigner function on a grid, W(x, k)
        
    k_vals : numpy array
        The k-vector values for the Wigner function grid, related to the x-values input by the FFT frequencies"""
        
    # get the values of the function in the shape of
    # the wigner function output
    f = func(x_vals)
    
    return _WignerTransformFFT(f, x_vals)

    
def WignerTransformArray(f_array, x_vals, t_vals, lambda_r):
    """Computes the 1D Wigner transform of an array on a grid
    
    Args:
    -----
    
    f_array : array
        Array of values to compute the Wigner function on. Must be the same shape as x_vals.
    
    x_vals : array
        The points in x where the function should be evaluated. Must be the same shape as f_array.
            
    k_vals : array
        The points in k-space where the function should be evaluated
        
    Returns:
    --------
    
    wig_func : numpy array
        The Wigner function on a grid
        
    k_vals : numpy array
        The k-vector values for the Wigner function grid, related to the x-values input by the FFT frequencies"""
    
    assert (np.shape(x_vals) == np.shape(f_array)), "f_array shape {} and x_val shape {} must be equal".format(np.shape(f_array),np.shape(x_val))
    
    k_vals = t_vals / lambda_r
    
    return _WignerTransform(f_array, x_vals)
    
    
def _WignerTransformFFT(f_array, x_vals):
    """Computes the Wigner transform from an array"""
    
    dx = x_vals[1] - x_vals[0]
    x_mat = np.matrix(x_vals)

    # Fourier transform the array
    f_tilde = fft(f_array)
    f_tilde_mat = np.tile(np.matrix(f_tilde).T, np.size(x_vals))
    
    # compute the k-values
    k_vals = np.matrix(fftfreq(n=np.size(f_tilde), d=dx))

    shift_phase  = np.exp(-PI*1j*k_vals.T*x_mat) ##divide by two to get phi/2 shift

    f_tilde_plus  = np.multiply(f_tilde_mat,shift_phase)
    f_tilde_minus = np.multiply(f_tilde_mat,np.conj(shift_phase))
    
    f_plus  = ifft(f_tilde_plus,n=None,axis=0)
    f_minus = np.conjugate(ifft(f_tilde_minus,n=None,axis=0))

    wig_arg = (
                (
                    np.multiply(f_plus,f_minus))
                ) # argument whose fourier transform is the wigner function

    wig_func = fft(fftshift(wig_arg, axes=1),n=None,axis=1,norm=None)
    wig_func = fftshift(wig_func,axes=1).T/dx
    
    return wig_func


def _WignerTransform(f_array, x_vals, q_vals):
    """Computes the Wigner transform from an array"""
    
    dx = x_vals[1] - x_vals[0]
    x_mat = np.matrix(x_vals)

    # Fourier transform the array
    f_tilde = fft(f_array)
    f_tilde_mat = np.tile(np.matrix(f_tilde).T, np.size(x_vals))
    
    # compute the k-values
    k_vals = np.matrix(fftfreq(n=np.size(f_tilde), d=dx))
    
    shift_phase  = np.exp(-PI*1j*k_vals.T*x_mat) ##divide by two to get phi/2 shift

    f_tilde_plus  = np.multiply(f_tilde_mat,shift_phase)
    f_tilde_minus = np.multiply(f_tilde_mat,np.conj(shift_phase))
    
    f_plus  = ifft(f_tilde_plus,n=None,axis=0)
    f_minus = np.conjugate(ifft(f_tilde_minus,n=None,axis=0))

    wig_arg = np.real(
                (
                    np.multiply(f_plus,f_minus).T)
                ) # argument whose fourier transform is the wigner function
    
    # generate the 3D array x, y, z
    fourier_array = np.exp(2.*np.pi*1.j*np.outer(x_vals, q_vals))
    integrand = np.einsum('yx, yq -> yxq', wig_arg, fourier_array)
    
    wigner_func = trapz(integrand, x_vals, axis=0).T
    
    return wigner_func
