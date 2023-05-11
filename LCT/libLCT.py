import math
#import srwlib
import numpy as np
from srwlib import *


def rotation_22(theta):
    """
    Return the 2x2 rotation matrix that rotates vectors in
    the plane counter-clockwise (CCW) by angle theta.
    """
    c, s = (np.cos(theta), np.sin(theta))
    return [[c, -s], [s, c]]


def permute_nn_22(arr):
    """
    Permute the entries of the given array according to
    the transformation from phase-space variables
      (q1,...,qn, p1,...,pn)
    to phase-space variables
      (q1,p1, ..., qn,pn).

    Vectors transform according to the rule
        v' = P.v,
    and matrices transform according to the rule
        M' = P.M.tr(P),
    where P denotes the apprpriate permutation matrix,
    and tr(P) denotes its transpose.
    """
    n2 = len(arr)
    assert n2 % 2 == 0, "Argument must be an even-length array."

    pmx = np.eye(n2)
    jq = np.arange(0, n2-1,  2)
    jp = np.arange(1,  n2,   2)
    pmx = pmx[ np.concatenate((jq, jp)) ].T

    if np.shape(arr) == (n2,):
        return np.matmul(pmx, arr)
    elif np.shape(arr) == (n2,n2):
        return np.matmul(pmx, np.matmul(arr, pmx.T))
    else:
        assert False, \
          'Argument is neither an even-length vector '\
          'nor an even-length square matrix!'


def permute_22_nn(arr):
    """
    Permute the entries of the given array according to
    the transformation from phase-space variables
      (q1,p1, ..., qn,pn)
    to phase-space variables
      (q1,...,qn, p1,...,pn).
    
    Vectors transform according to the rule
        v' = P.v,
    and matrices transform according to the rule
        M' = P.M.tr(P),
    where P denotes the apprpriate permutation matrix,
    and tr(P) denotes its transpose.
    """
    n2 = len(arr)
    assert n2 % 2 == 0, "Argument must be an even-length array."

    pmx = np.eye(n2)
    jq = np.arange(0, n2-1,  2)
    jp = np.arange(1,  n2,   2)
    pmx = pmx[ np.concatenate((jq, jp)) ]

    if np.shape(arr) == (n2,):
        return np.matmul(pmx, arr)
    elif np.shape(arr) == (n2,n2):
        return np.matmul(pmx, np.matmul(arr, pmx.T))
    else:
        assert False, \
          'Argument is neither an even-length vector '\
          'nor an even-length square matrix!'


def convert_params_3to4(alpha, beta, gamma):
    """
    Given LCT parameters (α,β,γ), return the associated 2x2 ABCD matrix.

    Caveats: Not all authors use the same convention for the parameters
    (α,β,γ): some reverse the rôles of α and γ.
    We follow the notation of [Koç [7]](#ref:Koc-2011-thesis)
    and [Koç, et al. [8]](#ref:Koc-2008-DCLCT),
    also [Healy [4], ch.10](#ref:Healy:2016:LCTs).

    Restrictions: The parameter beta may not take the value zero.

    Arguments:
    α, β, γ -- a parameter triplet defining a 1D LCT

    Return a symplectic 2x2 ABCD martrix.
    """
    if beta == 0.:
        print("The parameter beta may not take the value zero!")
        return -1
    M = np.zeros((2,2))
    M[0,0] = gamma / beta
    M[0,1] = 1. / beta
    M[1,0] = -beta + alpha * gamma / beta
    M[1,1] = alpha / beta

    return M


def convert_params_4to3(M_lct):
    """
    Given a symplectic 2x2 ABCD matrix, return the associated parameter triplet (α,β,γ).

    Caveats: Not all authors use the same convention for the parameters
    (α,β,γ): some reverse the rôles of α and γ.
    We follow the notation of [Koç [7]](#ref:Koc-2011-thesis)
    and [Koç, et al. [8]](#ref:Koc-2008-DCLCT),
    also [Healy [4], ch.10](#ref:Healy:2016:LCTs).

    Restrictions: The (1,2) matrix entry may not take the value zero.

 ** DTA: We need to decide how best to handle b very near zero.

    Arguments:
    M_lct -- a symplectic 2x2 ABCD martrix that defines a 1D LCT

    Return the parameter triplet α, β, γ.
    """
    a, b, c, d = np.asarray(M_lct).flatten()
    if b == 0.:
        print("The (1,2) matrix entry may not take the value zero!")
        return -1
    beta = 1 / b
    alpha = d / b
    gamma = a / b

    return alpha, beta, gamma


def lct_abscissae(nn, du, ishift = False):
    """
    Return abscissae for a signal of length nn with spacing du.

    With the default setting `ishift = False`, the abscissae will
    be either centered on the origin (if an odd number of points),
    or one step off of that (if an even number of points).
    If one sets `ishift = True`, then the computed array of abscissae
    will be shifted, or “rolled”, to place the origin at the beginning.

    Arguments:
    nn -- length of signal for which to construct abscissae,
            i.e. number of samples
    du -- sample spacing
    ishift -- a boolean argument: if True, the array of abcissae will
                be rotated so as to place the origin in entry [0]
    """
    u_vals = du * (np.arange(0, nn) - nn // 2)
    # if ishift == True: u_vals = np.roll(u_vals, (nn + 1) // 2)
    if ishift == True: u_vals = np.fft.ifftshift(u_vals)
    return u_vals


def resample_signal(k, in_signal, debug = False):
    """
    Resample the input signal, and return the resultant signal.

    This function takes an input signal
        U = [dt, [u0, u1, ..., u_{n-1}]]
    and resamples it by a factor of k, returning the new signal
        SMPL(k){U} = [dt', [u0, u1, ..., u_{n'-1}]],
    where n' is _roughly_ k * n. The “roughly” has to do with the
    fact that k need not be an integer.

    This function requires interpolating the data to a new sample
    interval: The initial range is Δt = n * dt, with sample points
    at the centers of n equal-size intervals, and the function is
    taken to have period Δt. The points for the resampled signal
    will occupy the _same_ range Δt.

 ** DTA: This function currently uses 1D _linear_ interpolation.
         We should upgrade this to use a more sophisticated
         interpolation scheme, such as those available in SciPy
         or that described by Pachón, et al.

    Arguments:
        k -- factor by which to resample the data
        in_signal -- [dt, [u_0, ..., u_{n-1}]], where dt denotes
                     the sample interval of the input signal [u]

    Return the resampled signal.
    """
    dt, signal_u = in_signal
    n = len(signal_u)

    # number of samples and sample spacing for resampled signal
    n_bar = int(np.ceil(k * n))
    dt_bar = dt * n / n_bar

    # abscissae
    t_vals = lct_abscissae(n,     dt    )
    t_bar  = lct_abscissae(n_bar, dt_bar)

    # interpolate signal at the new abscissae
    p = n * abs(dt)
    u_bar = np.interp(t_bar, t_vals, signal_u, period = p)

    if debug:
        print("n    = ", n, "   n_bar = ", n_bar)
        print("dt   =", dt, "  dt_bar =", dt_bar, "\n")
        print("t_in =", t_vals, "\n")
        print("t_up =", t_bar)

    return [dt_bar, u_bar]


def scale_signal(m, in_signal):
    """
    Scale the input signal, and return the result.

    This function implements the LCT version of 1D Scaling (SCL),
    with parameter m acting on a given input signal:
        SCL(m): LCT[m 0, 0 1/m]{f}(m.x) <- f(x) / sqrt(|m|).
    The input data must have the form
        [dX, [f_0, ..., f_{N-1}]],
    where dX denotes the sample interval of the incoming signal.

    Arguments:
    m -- scale factor
    in_signal -- the signal to transform, [dX, signal_array],
                 where dX denotes the incoming sample interval

    Return the scaled signal.
    """
    # NB: dX . ΔY = ΔX . dY = 1
    # and Ns = 1 + ΔX / dX = 1 + ΔY / dY
    dX, signalX = in_signal

    dY = abs(m) * dX
    signalY = np.copy(signalX) / np.sqrt(abs(m))
    if m < 0: # reverse signal
        ii = 0 + 1j  # “double-struck” i as unit imaginary
        signalY = ii * signalY[::-1]
        if len(signalY) % 2 == 0:
            # move what was the most negative frequency component,
            # now the most positive, back to the beginning
            signalY = np.roll(signalY, 1)

    return [dY, signalY]


def lct_fourier(in_signal):
    """
    Fourier transform the input signal, and return the result.

    This function implements the LCT version of a 1D Fourier transform (FT),
        FT(): LCT[0 1, -1 0]{f}(y) <- e^{-ii φ} FT(f),
    using numpy’s FFT to do the heavy lifting. As indicated here, the LCT
    version differs by an overall phase.

 ** DTA: KB Wolf remarks that correctly identifying the phase is a delicate
         matter. In light of that, we need to verify the phase used here.

    Argument:
    in_signal -- the signal to transform, [dX, signal_array], where dX
                 denotes the incoming sample interval, and we assume the
                 signal array is assumed symmetric (in the DFT sense)
                 about the origin

    Return the transformed signal in the form [dY, e^{-ii φ} FFT(signal)].
    """
    # NB: dX . ΔY = ΔX . dY = 1
    dX, signalX = in_signal
    Npts = len(signalX)
    dY = 1 / (Npts * dX)

    ii = 0 + 1j  # “double-struck” i as unit imaginary
    lct_coeff = np.exp(-ii * np.pi / 4) # DTA: KB Wolf says this requires care(!).

    # convert to frequency domain
    signalX = np.fft.ifftshift(signalX)
    signalY = dX * np.fft.fft(signalX)
    signalY = np.fft.fftshift(signalY)

    return [dY, lct_coeff * signalY]


def chirp_multiply(q, in_signal):
    """
    Transform the input signal by chirp multiplication with parameter q.

    This function implements the LCT version of chirp multiplication (CM)
    with parameter q acting on a given input signal:
        CM(q): LCT[1 0, q 1]{f}(x) <- e^{-ii π q x^2}f(x).
    The input data must have the form
        [dX, [f_0, ..., f_{N-1}]],
    where dX denotes the sample interval of the incoming signal.

    Arguments:
    q -- CM factor
    in_signal -- the signal to transform, [dX, signal_array],
                 where dX denotes the incoming sample interval

    Return the transformed signal.
    """
    # NB: dX . ΔY = ΔX . dY = 1
    dX, signalX = in_signal

    ii = 0 + 1j  # “double-struck” i as unit imaginary
    ptsX2 = lct_abscissae(len(signalX), dX) ** 2

    return [ dX, np.exp(-ii * np.pi * q * ptsX2) * signalX]


def lct_decomposition(M_lct):
    """
    Given an LCT matrix, M_lct, return a decomposition into simpler matrices. 

    Any symplectic 2x2 ABCD matrix that defines a linear canonical transform
    (LCT) may be decomposed as a product of matrices that each correspond to
    simpler transforms for which fast [i.e., ~O(N log N)] algorithms exist.
    The transforms required here are scaling (SCL), chirp multiplication (CM),
    and the Fourier transform (FT). In addition, we must sometimes resample
    the data (SMPL) so as to maintain a time-bandwidh product sufficient to
    recover the original signal.

 ** DTA: Must we handle separately the case B = M_lct[1,2] = 0?
         What about the case |B| << 1?

    The decompositions used here comes from the work of Koç, et al.,
    in _IEEE Trans. Signal Proc._ 56(6):2383--2394, June 2008.

    Argument:
    M_lct -- symplectic 2x2 matrix that describes the desired LCT

    Return an array of size mx2, where m denotes the total number of
    operations in the decomposition. Each row has the form ['STR', p],
    where 'STR' specifies the operation, and p the parameter relevant
    for that operation.
    """
    alpha, beta, gamma = convert_params_4to3(M_lct)
    ag = abs(gamma)
    if ag <= 1:
        k = 1 + ag + abs(alpha) / beta ** 2 * (1 + ag) ** 2
        seq = [ [ 'SCL',   beta              ],
                ['RSMP',     2.              ],
                [  'CM', - gamma / beta ** 2 ],
                ['LCFT',     0               ],
                ['RSMP',    k/2              ],
                [  'CM', - alpha             ] ]
    else:
        k = 1 + 1 / ag + abs(alpha - beta ** 2 / gamma) / beta ** 2 * (1 + ag) ** 2
        seq = [ [ 'SCL', - gamma / beta              ],
                ['LCFT',     0                       ],
                ['RSMP',     2.                      ],
                [  'CM',   gamma / beta ** 2         ],
                ['LCFT',     0                       ],
                ['RSMP',    k/2                      ],
                [  'CM', - alpha + beta ** 2 / gamma ] ]

    return seq


def lctmx_scale(m):
    return np.array([[ m,  0  ],
                     [ 0, 1/m ]])


def lctmx_chirp_x(q):
    return np.array([[  1, 0 ],
                     [ -q, 1 ]])


def lctmx_fourier():
    return np.array([[  0, 1 ],
                     [ -1, 0 ]])


def apply_lct(M_lct, in_signal):
    """
    Apply LCT[M_lct] to a given input signal, and return the result.

    Given a symplectic 2x2 ABCD matrix that defines an LCT, decompose
    the matrix into a sequence of simpler operations, so as to achieve
    an operation count of ~O(N log N).

    The algorithm implemented here is that given by Koç, et al.
    in IEEE Trans. Signal Proc. 56(6):2383--2394, June 2008.

 ** DTA: Consider implementing one or more of the other known
         fast LCT algorithms. Doing so can help with verifying
         correctness, as well as allow us to learn something
         about the relative performance of different algorithms.

    Arguments:
    M_lct -- symplectic 2x2 matrix that describes the desired LCT
    in_signal -- the signal to transform, [ dX, signal_array ], where
                 dX denotes the sample interval of the given signal

    Return the transformed signal in the form [ dY, LCT[M_lct](signal)].
    """
    seq = lct_decomposition(M_lct)
    signal0 = in_signal
    for lct in seq:
        if   lct[0] == 'CM':
            signal1 = chirp_multiply(lct[1], signal0)
            signal0 = signal1
        elif lct[0] == 'LCFT':
            signal1 = lct_fourier(signal0)
            signal0 = signal1
        elif lct[0] == 'SCL':
            signal1 = scale_signal(lct[1], signal0)
            signal0 = signal1
        elif lct[0] == 'RSMP':
            signal1 = resample_signal(lct[1], signal0)
            signal0 = signal1
        else:
            assert False, 'LCT code ' + lct[0] + ' not recognized! Exiting now.'
            return -1

    return signal1


def apply_lct_2d_sep(MX_lct, MY_lct, in_signal):
    """
    Apply LCT[M_lct] to a given input signal, and return the result.
    In this case, the 4x4 matrix M_lct is assumed separable and defined
    by the pair of 2x2 matrices MX_lct and MY_lct. In addition, the
    input signal is taken to have the form ( dX, dY, signal_array ),
    where dX and dY denote the sample intervals along the rows and
    columns respectively of the given 2D signal_array.

    Given a pair of symplectic 2x2 ABCD matrices that define an
    uncoupled LCT in two dimensions, apply LCT[MX_lct] to each row,
    and then LCT[MY_lct] to each column.

    The algorithm implemented here is that given by Koç, et al.
    in IEEE Trans. Signal Proc. 56(6):2383--2394, June 2008.

    Arguments:
    MX_lct -- symplectic 2x2 matrix that describes the desired LCT
    MY_lct -- symplectic 2x2 matrix that describes the desired LCT
    in_signal -- the signal to transform, [ dX, dY, signal_array], where
                 dX and dY denote the sample intervals of the given
                 signal along its two axes

    Return the transformed signal in the form [ dY, LCT[M_lct](signal)].
    """
    # extract the pieces
    dX, dY, signal_array = in_signal

    # apply LCT[MX_lct] to each row
    lct_x = [ apply_lct(MX_lct, (dX, sig_x)) for sig_x in signal_array  ]
    dX_out = lct_x[0][0]

    # apply LCT[MY_lct] to each column
    signal_array = np.asarray([ s[-1] for s in lct_x ]).T
    lct_y = [ apply_lct(MY_lct, (dY, sig_y)) for sig_y in signal_array ]
    dY_out = lct_y[0][0]

    signal_array = np.asarray([ s[-1] for s in lct_y ]).T
    return (dX_out, dY_out, signal_array)




# SRW-related LCT functions:

def createABCDbeamline(A,B,C,D):
    """
    #Use decomposition of ABCD matrix into kick-drift-kick Pei-Huang 2017 (https://arxiv.org/abs/1709.06222)
    #Construct corresponding SRW beamline container object
    #A,B,C,D are 2x2 matrix components.
    """
    
    f1= B/(1-A)
    L = B
    f2 = B/(1-D)
    
    optLens1 = SRWLOptL(f1, f1)
    optDrift = SRWLOptD(L)
    optLens2 = SRWLOptL(f2, f2)
    
    propagParLens1 = [0, 0, 1., 0, 0, 1, 1, 1, 1, 0, 0, 0]
    propagParDrift = [0, 0, 1., 0, 0, 1, 1, 1, 1, 0, 0, 0]
    propagParLens2 = [0, 0, 1., 0, 0, 1, 1, 1, 1, 0, 0, 0]
    
    optBL = SRWLOptC([optLens1,optDrift,optLens2],[propagParLens1,propagParDrift,propagParLens2])
    return optBL


def createGsnSrcSRW(sigrW,propLen,pulseE,poltype,phE=10e3,sampFact=15,mx=0,my=0):
    """
    #sigrW: beam size at waist [m]
    #propLen: propagation length [m] required by SRW to create numerical Gaussian
    #pulseE: energy per pulse [J]
    #poltype: polarization type (0=linear horizontal, 1=linear vertical, 2=linear 45 deg, 3=linear 135 deg, 4=circular right, 5=circular left, 6=total)
    #phE: photon energy [eV]
    #sampFact: sampling factor to increase mesh density
    """
    
    constConvRad = 1.23984186e-06/(4*3.1415926536)  ##conversion from energy to 1/wavelength
    rmsAngDiv = constConvRad/(phE*sigrW)             ##RMS angular divergence [rad]
    sigrL=math.sqrt(sigrW**2+(propLen*rmsAngDiv)**2)  ##required RMS size to produce requested RMS beam size after propagation by propLen
    
        
    #***********Gaussian Beam Source
    GsnBm = SRWLGsnBm() #Gaussian Beam structure (just parameters)
    GsnBm.x = 0 #Transverse Positions of Gaussian Beam Center at Waist [m]
    GsnBm.y = 0
    GsnBm.z = propLen #Longitudinal Position of Waist [m]
    GsnBm.xp = 0 #Average Angles of Gaussian Beam at Waist [rad]
    GsnBm.yp = 0
    GsnBm.avgPhotEn = phE #Photon Energy [eV]
    GsnBm.pulseEn = pulseE #Energy per Pulse [J] - to be corrected
    GsnBm.repRate = 1 #Rep. Rate [Hz] - to be corrected
    GsnBm.polar = poltype #1- linear horizontal?
    GsnBm.sigX = sigrW #Horiz. RMS size at Waist [m]
    GsnBm.sigY = GsnBm.sigX #Vert. RMS size at Waist [m]

    GsnBm.sigT = 10e-15 #Pulse duration [s] (not used?)
    GsnBm.mx = mx #Transverse Gauss-Hermite Mode Orders
    GsnBm.my = my

    #***********Initial Wavefront
    wfr = SRWLWfr() #Initial Electric Field Wavefront
    wfr.allocate(1, 1000, 1000) #Numbers of points vs Photon Energy (1), Horizontal and Vertical Positions (dummy)
    wfr.mesh.zStart = 0.0 #Longitudinal Position [m] at which initial Electric Field has to be calculated, i.e. the position of the first optical element
    wfr.mesh.eStart = GsnBm.avgPhotEn #Initial Photon Energy [eV]
    wfr.mesh.eFin = GsnBm.avgPhotEn #Final Photon Energy [eV]

    wfr.unitElFld = 1 #Electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2), 2- sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on representation (freq. or time)

    distSrc = wfr.mesh.zStart - GsnBm.z
    #Horizontal and Vertical Position Range for the Initial Wavefront calculation
    #can be used to simulate the First Aperture (of M1)
    #firstHorAp = 8.*rmsAngDiv*distSrc #[m]
    xAp = 6.*sigrL
    yAp = xAp #[m]
    
    wfr.mesh.xStart = -0.5*xAp #Initial Horizontal Position [m]
    wfr.mesh.xFin = 0.5*xAp #Final Horizontal Position [m]
    wfr.mesh.yStart = -0.5*yAp #Initial Vertical Position [m]
    wfr.mesh.yFin = 0.5*yAp #Final Vertical Position [m]
    
    sampFactNxNyForProp = sampFact #sampling factor for adjusting nx, ny (effective if > 0)
    arPrecPar = [sampFactNxNyForProp]
    
    srwl.CalcElecFieldGaussian(wfr, GsnBm, arPrecPar)
    
    ##Beamline to propagate to waist
    
    optDriftW=SRWLOptD(propLen)
    propagParDrift = [0, 0, 1., 1, 0, 1.1, 1.2, 1.1, 1.2, 0, 0, 0]
    optBLW = SRWLOptC([optDriftW],[propagParDrift])
    #wfrW=deepcopy(wfr)
    srwl.PropagElecField(wfr, optBLW)
    
    return wfr
