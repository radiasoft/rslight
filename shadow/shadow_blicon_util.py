import math
import numpy as np
import Shadow
from Shadow.ShadowPreprocessorsXraylib import prerefl, pre_mlayer, bragg
from srxraylib.sources import srfunc
import matplotlib.pyplot as plt

def rays_sigma(beam, idx, idy):
    rays = beam.rays
    return np.sqrt(np.mean(rays[:, idx] * rays[:, idy]))

def shdw_plt(beam):
    """
    This function plots xy intensity
    distribution from Shadow beam
    object.
    """
    xvals_full_beam = beam.getshonecol(col = 1)
    yvals_full_beam = beam.getshonecol(col = 3)
    nbins_shadow_int = 100
    data_full_beam = beam.histo2(col_h = 3, col_v = 1, nbins = nbins_shadow_int, ref=23, nbins_h=None, nbins_v=None, nolost=0, xrange=None, yrange=None, calculate_widths=1)
    data_beam = data_full_beam[('histogram')]
    xvals_beam = np.linspace(np.min(xvals_full_beam), np.max(xvals_full_beam), nbins_shadow_int)
    yvals_beam = np.linspace(np.min(yvals_full_beam), np.max(yvals_full_beam), nbins_shadow_int)
    # printmd('$\Delta x$ = %s $mm$' %(x_off_e0*1e3))
    fig = plt.figure(figsize=(8,6))
    ax = fig.gca()
    plt.pcolormesh(np.multiply(xvals_beam,10), np.multiply(yvals_beam,10), data_beam, cmap=plt.cm.Blues)
    plt.colorbar()
    #ax.set_xlim(-20,20)
    #ax.set_ylim(-yrange/2,yrange/2)
    ax.set_ylabel(r'Y [mm]')
    ax.set_xlabel(r'X [mm]')
    # ax.set_title('Intensity',**hfontMed)
    ax.grid()
    plt.show()


def shadow_src_beam(n_rays=10000, ran_seed=15829, dist_type=3, sigx=4.42e-05, sigz=2.385e-05, sigdix=3.52e-05, sigdiz = 2.875e-05, hdiv1 = 0.0, hdiv2 = 0.0, vdiv1 =0.0, vdiv2=0.0, ph_energy = 1e3):
    """
    This function computes a shadow
    beam object from a specified source.
    """

    source = Shadow.Source()
    beam = Shadow.Beam()

    source.NPOINT = n_rays # no. of rays (1 on-axis ray, 4 deviations)
    source.ISTAR1 = ran_seed

    #source.FSOUR = src_type  # source type (0 = point, 3 = Gsn)
    #source.WXSOU = wx
    #source.WZSOU = wz
    source.SIGMAX = sigx
    source.SIGMAZ = sigz
    source.FDISTR = dist_type
    source.SIGDIX = sigdix
    source.SIGDIZ = sigdiz
    source.F_POLAR = 0
    source.HDIV1 = hdiv1
    source.HDIV2 = hdiv2
    source.VDIV1 = vdiv1
    source.VDIV2 = vdiv2
    source.IDO_VX = 0
    source.IDO_VZ = 0
    source.IDO_X_S = 0
    source.IDO_Y_S = 0
    source.IDO_Z_S = 0
    source.F_PHOT = 0
    source.PH1 = ph_energy
    beam.genSource(source)

    return beam


def run_shdw_kb_gsn(beam):

    """
    beam: initial set of rays from genSource
    """



    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.FMIRR = 2
    oe.ALPHA = 90
    oe.FHIT_C = 1
    oe.F_EXT = 0
    oe.F_DEFAULT = 0
    oe.SSOUR = 2850.0
    oe.SIMAG = 900.0
    oe.THETA = 87.99464771704211
    oe.F_CONVEX = 0
    oe.FCYL = 1
    oe.CIL_ANG = 0.0
    oe.FSHAPE = 1
    oe.RWIDX1 = 0.5
    oe.RWIDX2 = 0.5
    oe.RLEN1 = 25.0
    oe.RLEN2 = 25.0
    oe.T_INCIDENCE = 87.99464771704211
    oe.T_REFLECTION = 87.99464771704211
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 2850.0
    beam.traceOE(oe, 1)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty(ALPHA=270)
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 0.0
    beam.traceOE(oe, 2)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.FMIRR = 2
    oe.ALPHA = 0
    oe.FHIT_C = 1
    oe.F_EXT = 0
    oe.F_DEFAULT = 0
    oe.SSOUR = 2950.0
    oe.SIMAG = 800.0
    oe.THETA = 87.99464771704211
    oe.F_CONVEX = 0
    oe.FCYL = 1
    oe.CIL_ANG = 0.0
    oe.FSHAPE = 1
    oe.RWIDX1 = 0.5
    oe.RWIDX2 = 0.5
    oe.RLEN1 = 25.0
    oe.RLEN2 = 25.0
    oe.T_INCIDENCE = 87.99464771704211
    oe.T_REFLECTION = 87.99464771704211
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 100.0
    beam.traceOE(oe, 3)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty()
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 800.0
    beam.traceOE(oe, 4)
    beam.write('shadow-output.dat')
    
    return beam

    