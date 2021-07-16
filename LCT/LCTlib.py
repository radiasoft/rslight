import math
#import srwlib
import numpy as np
from srwlib import *

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
