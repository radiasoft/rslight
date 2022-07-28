def main(): 
  
  """
  x = np.linspace(0.0, 6.0, 121) 
  ys = spec.erf(x) 
  ya = approx_erf(x) 
  yg = np.sqrt( 1. -1. /(1. +x *x) ) 
  plt.plot(x, ys, 'b-', label=r'hard-edge (exact)') 
  plt.plot(x, ya, 'r--', label=r'hard-edge (approx.)') 
  plt.plot(x, yg, 'g--', label=r'Gaussian w/ $a_g = a_h$') 
  plt.xlim(xmin=0.0) 
  plt.ylim(ymin=0.0) 
  plt.xlabel(r'$a /\sqrt{2}\sigma_x$') 
  plt.ylabel(r'Gain')
  plt.legend()
  #plt.savefig('gain.pdf') 
  plt.show()  
  
  ah = np.linspace(.0001, 3.0001, 301) 
  ag = ag_from_ah(ah) 
  plt.plot(ah, ah, 'r', label=r'$a_h$ (hard-edge)') 
  plt.plot(ah, ag, 'b', label=r'$a_g$ (Gaussian)') 
  plt.xlim(xmin=0.0) 
  plt.xlabel(r'$a_h /\sqrt{2}\sigma_x$') 
  plt.ylabel(r'$a /\sqrt{2}\sigma_x$')
  plt.legend() 
  #plt.savefig('ah_and_ag.pdf') 
  plt.show() 
  
  plt.plot(ah, ag/ah) 
  plt.xlim(xmin=0.0) 
  plt.ylim(ymin=0.0) 
  plt.xlabel(r'$a_h /\sqrt{2}\sigma_x$') 
  plt.ylabel(r'$a_g / a_h$')
  #plt.savefig('ag_over_ah.pdf')
  plt.show() 
  """

  #sig_test = np.array( [[11, 12, 13, 14], [12, 22, 23, 24], [13, 23, 33, 34], [14, 24, 34, 44]] ) 
  sig_test = np.array( [[11, 0, 13, 0], [0, 22, 0, 24], [13, 0, 33, 0], [0, 24, 0, 44]] ) 
  print sig_test  #  This is Sigma, not inv(Sigma) 
  lambda_rad = 1.0 
  a_gx = 1.0 
  a_gy = 2.0 
  sig_proped = gauss_apert_4x4(sig_test, lambda_rad, a_gx, a_gy) 
  print sig_proped 
  
  # Cross-check __for the case of x-y separable__ distribution (ordering: x, y, theta_x, theta_y) 
  sig_xx_f = 0.5 *a_gx**2 *sig_test[0,0] /(0.5 *a_gx**2 +sig_test[0,0]) 
  sig_xtx_f = 0.5 *a_gx**2 *sig_test[0,2] /(0.5 *a_gx**2 +sig_test[0,0]) 
  sig_txtx_f = sig_test[2,2] -sig_test[0,2] *sig_test[0,2] /(0.5 *a_gx**2 +sig_test[0,0]) +lambda_rad**2 /(
    8. *np.pi *np.pi *a_gx**2)
  
  sig_yy_f = 0.5 *a_gy**2 *sig_test[1,1] /(0.5 *a_gy**2 +sig_test[1,1]) 
  sig_yty_f = 0.5 *a_gy**2 *sig_test[1,3] /(0.5 *a_gy**2 +sig_test[1,1]) 
  sig_tyty_f = sig_test[3,3] -sig_test[1,3] *sig_test[1,3] /(0.5 *a_gy**2 +sig_test[1,1]) +lambda_rad**2 /(
    8. *np.pi *np.pi *a_gy**2)
  
  sig_proped_sep = np.zeros([4,4], dtype=np.float32) 
  
  sig_proped_sep[0,0] = sig_xx_f 
  sig_proped_sep[0,2] = sig_xtx_f 
  sig_proped_sep[2,0] = sig_xtx_f 
  sig_proped_sep[2,2] = sig_txtx_f 
  
  sig_proped_sep[1,1] = sig_yy_f 
  sig_proped_sep[1,3] = sig_yty_f 
  sig_proped_sep[3,1] = sig_yty_f 
  sig_proped_sep[3,3] = sig_tyty_f 
  
  print ' '
  print sig_proped_sep 
  print ' '





def approx_erf(x): 
  a1 = 0.278393 
  a2 = 0.230389 
  a3 = 0.000972 
  a4 = 0.078108 
  
  signx = np.sign(x) 
  x = np.abs(x) 
  x = 1 +(a1 +(a2 +(a3 +a4 *x) *x) *x) *x 
  x = x *x *x *x  # np.power(x, 4) 
  return signx *(1. -1. /x)  

#  Effective Gaussian aperture size from the _hard-edge_ aperture size 
def ag_from_ah(ah): 
  G_h = spec.erf(ah) 
  ag = G_h /np.sqrt(1. -G_h *G_h) 
  return ag 


#  Fractional flux loss in passing through an a_x by a_y hard-edge rectangular aperture (centered beam)
def flux_loss_4Dcentered(a_x, a_y, sigma_x, sigma_y): 
  G_x = spec.erf(a_x /(np.sqrt(2.)*sigma_x)) 
  G_y = spec.erf(a_y /(np.sqrt(2.)*sigma_y)) 
  return G_x *G_y 

#  A faster version of the flux_loss_4Dcentered
def flux_loss_4Dcentered_appr(a_x, a_y, sigma_x, sigma_y): 
  G_x = approx_erf(a_x /(np.sqrt(2.)*sigma_x)) 
  G_y = approx_erf(a_y /(np.sqrt(2.)*sigma_y)) 
  return G_x *G_y 


#  Propagate a 4x4 covariance matrix Sigma through a Gaussian aperture of (Gaussian, not not hard-edge)
#  size parameters a_gx, a_gy 
#  NB:  assumed ordering of the variables is x, y, theta_x, theta_y 
def gauss_apert_4x4(Sigma, lambda_rad, a_gx, a_gy): 
  Sigma_inv = sla.inv(Sigma) 
  A = 1.0 *Sigma_inv[0:2,0:2] 
  B = 1.0 *Sigma_inv[0:2,2:4] 
  C = np.transpose(B) 
  D = 1.0 *Sigma_inv[2:4,2:4] 
  
  A_A = np.zeros([2,2], dtype=np.float32) 
  A_A[0,0] = 2. /a_gx**2 
  A_A[1,1] = 2. /a_gy**2 

  D_inv = sla.inv(D) 
  D_A_inv = np.zeros([2,2], dtype=np.float32) 
  D_A_inv[0,0] = 1. /a_gx**2 
  D_A_inv[1,1] = 1. /a_gy**2 
  D_A_inv *= lambda_rad**2 /(8. *np.pi *np.pi) 

  D_f = sla.inv(D_inv +D_A_inv) 
  BDi = np.matmul(B, D_inv) 
  DiC = np.matmul(D_inv, C)  #  == np.transpose(BDi) 
  C_f = np.matmul(D_f, DiC) 
  B_f = np.transpose(C_f)    #  ==  np.matmul(BDi, D_f) 
  A_f = A +A_A -np.matmul(BDi, C) +np.matmul(BDi, np.matmul(D_f, DiC)) 

  Sigma_inv[0:2,0:2] = 1.0 *A_f 
  Sigma_inv[0:2,2:4] = 1.0 *B_f 
  Sigma_inv[2:4,0:2] = 1.0 *C_f 
  Sigma_inv[2:4,2:4] = 1.0 *D_f 

  return sla.inv(Sigma_inv) 





if __name__=="__main__":
  import matplotlib.pyplot as plt 
  import numpy as np 
  import scipy.special as spec 
  import scipy.linalg as sla 
  import time 
  main() 


















