import math
import numpy as np
import Shadow
from Shadow.ShadowPreprocessorsXraylib import prerefl, pre_mlayer, bragg
from srxraylib.sources import srfunc
import matplotlib.pyplot as plt

def rays_sigma(beam, idx, idy):
    rays = beam.rays
    return np.sqrt(np.mean(rays[:, idx] * rays[:, idy]))

def rays_sigma_mat(beam):
    """
    This function takes a Shadow beam object
    and returns a Sigma matrix based on the
    input distribution.
    """
    
    rays = beam.rays
    sigmaxx = np.mean(rays[:, 0] * rays[:, 0])
    sigmaxxp = np.mean(rays[:, 0] * rays[:, 3])
    sigmaxz = np.mean(rays[:, 0] * rays[:, 2])
    sigmaxzp = np.mean(rays[:, 0] * rays[:, 5])
    
    sigmaxpx = np.mean(rays[:, 3] * rays[:, 0])
    sigmaxpxp = np.mean(rays[:, 3] * rays[:, 3])
    sigmaxpz = np.mean(rays[:, 3] * rays[:, 2])
    sigmaxpzp = np.mean(rays[:, 3] * rays[:, 5])
    
    sigmazx = np.mean(rays[:, 2] * rays[:, 0])
    sigmazxp = np.mean(rays[:, 2] * rays[:, 3])
    sigmazz = np.mean(rays[:, 2] * rays[:, 2])
    sigmazzp = np.mean(rays[:, 2] * rays[:, 5])
    
    sigmazpx = np.mean(rays[:, 5] * rays[:, 0])
    sigmazpxp = np.mean(rays[:, 5] * rays[:, 3])
    sigmazpz = np.mean(rays[:, 5] * rays[:, 2])
    sigmazpzp = np.mean(rays[:, 5] * rays[:, 5])
    
    sigma_mat = np.matrix([
        [sigmaxx, sigmaxxp, sigmaxz, sigmaxzp],
        [sigmaxpx, sigmaxpxp, sigmaxpz, sigmaxpzp],
        [sigmazx, sigmazxp, sigmazz, sigmazzp],
        [sigmazpx, sigmazpxp, sigmazpz, sigmazpzp],
    ])
    
    return sigma_mat

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



def run_shdw_tes(beam):
    
    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty().set_screens()
    oe.I_SLIT[0] = 1
    oe.K_SLIT[0] = 0
    oe.I_STOP[0] = 0
    oe.RX_SLIT[0] = 3.0
    oe.RZ_SLIT[0] = 0.05
    oe.CX_SLIT[0] = 0.0
    oe.CZ_SLIT[0] = 0.0
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 1050.0
    beam.traceOE(oe, 1)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.FMIRR = 1
    oe.ALPHA = 0
    oe.FHIT_C = 1
    oe.F_EXT = 1
    oe.F_CONVEX = 0
    oe.FCYL = 1
    oe.CIL_ANG = 0.0
    oe.RMIRR = 270000.0
    oe.FSHAPE = 1
    oe.RWIDX1 = 4000.0
    oe.RWIDX2 = 4000.0
    oe.RLEN1 = 40650.0
    oe.RLEN2 = 40650.0
    oe.T_INCIDENCE = 89.61038869931105
    oe.T_REFLECTION = 89.61038869931105
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 50.0
    beam.traceOE(oe, 2)

#     oe = Shadow.OE()
#     oe.DUMMY = 1.0
#     oe.FMIRR = 5
#     oe.ALPHA = 0
#     oe.FHIT_C = 0
#     oe.F_CRYSTAL = 1
#     oe.F_CENTRAL = 1
#     oe.F_PHOT_CENT = 0
#     oe.PHOT_CENT = 2500.0
#     oe.F_REFRAC = 0
#     oe.F_MOSAIC = 0
#     oe.F_BRAGG_A = 0
#     oe.F_JOHANSSON = 0
#     bragg(interactive=False, DESCRIPTOR='Si', H_MILLER_INDEX=1, K_MILLER_INDEX=1, L_MILLER_INDEX=1, TEMPERATURE_FACTOR=1.0, E_MIN=2000.0, E_MAX=3000.0, E_STEP=50.0, SHADOW_FILE='crystal-bragg-3.txt')
#     oe.FILE_REFL = b'crystal-bragg-3.txt'
#     oe.FWRITE = 3
#     oe.T_IMAGE = 0.0
#     oe.T_SOURCE = 1400.0
#     beam.traceOE(oe, 3)

#     oe = Shadow.OE()
#     oe.DUMMY = 1.0
#     oe.FMIRR = 5
#     oe.ALPHA = 180
#     oe.FHIT_C = 0
#     oe.F_CRYSTAL = 1
#     oe.F_CENTRAL = 1
#     oe.F_PHOT_CENT = 0
#     oe.PHOT_CENT = 2500.0
#     oe.F_REFRAC = 0
#     oe.F_MOSAIC = 0
#     oe.F_BRAGG_A = 0
#     oe.F_JOHANSSON = 0
#     bragg(interactive=False, DESCRIPTOR='Si', H_MILLER_INDEX=1, K_MILLER_INDEX=1, L_MILLER_INDEX=1, TEMPERATURE_FACTOR=1.0, E_MIN=2000.0, E_MAX=3000.0, E_STEP=50.0, SHADOW_FILE='crystal-bragg-4.txt')
#     oe.FILE_REFL = b'crystal-bragg-4.txt'
#     oe.FWRITE = 3
#     oe.T_IMAGE = 0.0
#     oe.T_SOURCE = 0.0
#     beam.traceOE(oe, 4)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty(ALPHA=180)
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 0.0
    beam.traceOE(oe, 5)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty()
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 157.0
    beam.traceOE(oe, 6)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.FMIRR = 3
    oe.ALPHA = 0
    oe.FHIT_C = 1
    oe.F_EXT = 1
    oe.F_TORUS = 0
    oe.R_MAJ = 2450000.0
    oe.R_MIN = 18.6
    oe.FSHAPE = 1
    oe.RWIDX1 = 4.0
    oe.RWIDX2 = 4.0
    oe.RLEN1 = 48.0
    oe.RLEN2 = 48.0
    oe.T_INCIDENCE = 89.59892954340842
    oe.T_REFLECTION = 89.59892954340842
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 0.0
    beam.traceOE(oe, 7)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty()
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 0.0
    beam.traceOE(oe, 8)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty()
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 2467.2
    beam.traceOE(oe, 9)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty()
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 190.0000000000009
    beam.traceOE(oe, 10)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty().set_screens()
    oe.I_SLIT[0] = 1
    oe.K_SLIT[0] = 0
    oe.I_STOP[0] = 0
    oe.RX_SLIT[0] = 0.02
    oe.RZ_SLIT[0] = 0.02
    oe.CX_SLIT[0] = 0.0
    oe.CZ_SLIT[0] = 0.0
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 0.0
    beam.traceOE(oe, 11)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty()
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 0.0
    beam.traceOE(oe, 12)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.FMIRR = 2
    oe.ALPHA = 90
    oe.FHIT_C = 1
    oe.F_EXT = 0
    oe.F_DEFAULT = 0
    oe.SSOUR = 357.3
    oe.SIMAG = 42.699999999999996
    oe.THETA = 89.79373519375291
    oe.F_CONVEX = 0
    oe.FCYL = 1
    oe.CIL_ANG = 0.0
    oe.FSHAPE = 1
    oe.RWIDX1 = 1.0
    oe.RWIDX2 = 1.0
    oe.RLEN1 = 16.0
    oe.RLEN2 = 16.0
    oe.T_INCIDENCE = 89.79373519375291
    oe.T_REFLECTION = 89.79373519375291
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 357.2999999999993
    beam.traceOE(oe, 13)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty(ALPHA=270)
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 0.0
    beam.traceOE(oe, 14)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.FMIRR = 2
    oe.ALPHA = 0
    oe.FHIT_C = 1
    oe.F_EXT = 0
    oe.F_DEFAULT = 0
    oe.SSOUR = 383.0
    oe.SIMAG = 17.0
    oe.THETA = 89.79373519375291
    oe.F_CONVEX = 0
    oe.FCYL = 1
    oe.CIL_ANG = 0.0
    oe.FSHAPE = 1
    oe.RWIDX1 = 1.0
    oe.RWIDX2 = 1.0
    oe.RLEN1 = 9.0
    oe.RLEN2 = 9.0
    oe.T_INCIDENCE = 89.79373519375291
    oe.T_REFLECTION = 89.79373519375291
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 25.699999999999818
    beam.traceOE(oe, 15)

    oe = Shadow.OE()
    oe.DUMMY = 1.0
    oe.set_empty()
    oe.FWRITE = 3
    oe.T_IMAGE = 0.0
    oe.T_SOURCE = 17.00000000000091
    beam.traceOE(oe, 16)
    beam.write('shadow-output.dat')

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

    