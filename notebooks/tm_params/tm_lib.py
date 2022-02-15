from __future__ import print_function
import uti_plot_com as srw_io
import numpy as np
import math
from math import sqrt
import re
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

import srwl_bl
import srwlib
import srwlpy
from srwlib import *
#from srwl_uti_brightness import *

#File processing functions
def read_srw_file(filename):
    data, mode, ranges, labels, units = srw_io.file_load(filename)
    data = np.array(data).reshape((ranges[8], ranges[5]), order='C')
    return {'data': data,
            'shape': data.shape,
            'mean': np.mean(data),
            'photon_energy': ranges[0],
            'horizontal_extent': ranges[3:5],
            'vertical_extent': ranges[6:8],
            # 'mode': mode,
            'labels': labels,
            'units': units}

def rmsfile(file):
    flux0=read_srw_file(file)
    data2D=flux0['data']
    datax=np.sum(data2D,axis=1) 
    datay=np.sum(data2D,axis=0)
    hx = flux0['horizontal_extent']
    hy = flux0['vertical_extent']
    shape=flux0['shape']
    xmin=hx[0]
    xmax=hx[1]
    ymin=hy[0]
    ymax=hy[1]
    Nx=shape[0]
    Ny=shape[1]
    dx = (xmax-xmin)/Nx
    dy = (ymax-ymin)/Ny
    xvals = np.linspace(xmin,xmax,Nx) 
    yvals = np.linspace(ymin,ymax,Ny)
    sxsq=sum(datax*xvals*xvals)/sum(datax) 
    xavg=sum(datax*xvals)/sum(datax)
    sx=sqrt(sxsq-xavg*xavg)

    sysq=sum(datay*yvals*yvals)/sum(datay) 
    yavg=sum(datay*yvals)/sum(datay)
    sy=sqrt(sysq-yavg*yavg)
    return sx, sy

#Defining general constants
_ElMass_kg=9.10938e-31
_Elch=1.60217662e-19
_LightSp=299792458
_ElMass_MeV=0.5109989
_Planck_eVs=4.135667662e-15



#!/usr/bin/env python
import os
try:
    __IPYTHON__
    import sys
    del sys.argv[1:]
except:
    pass

import srwl_bl
import srwlib
import srwlpy
import math
import srwl_uti_smp

def set_optics(v, names=None, want_final_propagation=True):
    el = []
    pp = []
    if not names:
        names = ['zero_drift', 'Watchpoint', 'Watchpoint_Toroid', 'Toroid', 'Toroid_Watchpoint2', 'Watchpoint2']
    for el_name in names:
        if el_name == 'zero_drift':
            # zero_drift: drift 24.99m
            el.append(srwlib.SRWLOptD(
                _L=v.op_zero_drift_L,
            ))
            pp.append(v.op_zero_drift_pp)
        elif el_name == 'Watchpoint':
            # Watchpoint: watch 24.99m
            pass
        elif el_name == 'Watchpoint_Toroid':
            # Watchpoint_Toroid: drift 24.99m
            el.append(srwlib.SRWLOptD(
                _L=v.op_Watchpoint_Toroid_L,
            ))
            pp.append(v.op_Watchpoint_Toroid_pp)
        elif el_name == 'Toroid':
            # Toroid: toroidalMirror 25.0m
            el.append(srwlib.SRWLOptMirTor(
                _rt=v.op_Toroid_rt,
                _rs=v.op_Toroid_rs,
                _size_tang=v.op_Toroid_size_tang,
                _size_sag=v.op_Toroid_size_sag,
                _x=v.op_Toroid_horizontalPosition,
                _y=v.op_Toroid_verticalPosition,
                _ap_shape=v.op_Toroid_ap_shape,
                _nvx=v.op_Toroid_nvx,
                _nvy=v.op_Toroid_nvy,
                _nvz=v.op_Toroid_nvz,
                _tvx=v.op_Toroid_tvx,
                _tvy=v.op_Toroid_tvy,
            ))
            pp.append(v.op_Toroid_pp)

        elif el_name == 'Toroid_Watchpoint2':
            # Toroid_Watchpoint2: drift 25.0m
            el.append(srwlib.SRWLOptD(
                _L=v.op_Toroid_Watchpoint2_L,
            ))
            pp.append(v.op_Toroid_Watchpoint2_pp)
        elif el_name == 'Watchpoint2':
            # Watchpoint2: watch 50.0m
            pass
    if want_final_propagation:
        pp.append(v.op_fin_pp)

    return srwlib.SRWLOptC(el, pp)

def epilogue():
    pass


def main(varParam):
    v = srwl_bl.srwl_uti_parse_options(srwl_bl.srwl_uti_ext_options(varParam), use_sys_argv=True)
    names = ['zero_drift','Watchpoint','Watchpoint_Toroid','Toroid','Toroid_Watchpoint2','Watchpoint2']
    op = set_optics(v, names, True)
    v.ws = True  # propagated single e-
    v.ws_pl = 'xy'
    v.si = True  # non-propagated single e-
    # v.si_pl = 'xy'
    srwl_bl.SRWLBeamline(_name=v.name).calc_all(v, op)

def tm_param(ws_fni, tm_x = 0.0, tm_y = 0.0, tm_nvx = 0.9999755001000415, tm_nvy = 0.0, tm_nvz = -0.006999942833473391, tm_tvx = 0.006999942833473391, tm_tvy = 0.0):
# def tm_param(tm_x = 0.0, tm_y = 0.0, ws_fni = 'res_int_pr_se.dat'):
    # tm_x: horizontal position  [m]?
    # tm_y: vertical position    [m]?

    
    varParam = [
        ['name', 's', 'tm01', 'simulation name'],

    #---Data Folder
        ['fdir', 's', '', 'folder (directory) name for reading-in input and saving output data files'],


        ['gbm_x', 'f', 0.0, 'average horizontal coordinates of waist [m]'],
        ['gbm_y', 'f', 0.0, 'average vertical coordinates of waist [m]'],
        ['gbm_z', 'f', 0.0, 'average longitudinal coordinate of waist [m]'],
        ['gbm_xp', 'f', 0.0, 'average horizontal angle at waist [rad]'],
        ['gbm_yp', 'f', 0.0, 'average verical angle at waist [rad]'],
        ['gbm_ave', 'f', 9000.0, 'average photon energy [eV]'],
        ['gbm_pen', 'f', 0.001, 'energy per pulse [J]'],
        ['gbm_rep', 'f', 1, 'rep. rate [Hz]'],
        ['gbm_pol', 'f', 1, 'polarization 1- lin. hor., 2- lin. vert., 3- lin. 45 deg., 4- lin.135 deg., 5- circ. right, 6- circ. left'],
        ['gbm_sx', 'f', 9.787229999999999e-06, 'rms beam size vs horizontal position [m] at waist (for intensity)'],
        ['gbm_sy', 'f', 9.787229999999999e-06, 'rms beam size vs vertical position [m] at waist (for intensity)'],
        ['gbm_st', 'f', 1e-13, 'rms pulse duration [s] (for intensity)'],
        ['gbm_mx', 'f', 0, 'transverse Gauss-Hermite mode order in horizontal direction'],
        ['gbm_my', 'f', 0, 'transverse Gauss-Hermite mode order in vertical direction'],
        ['gbm_ca', 's', 'c', 'treat _sigX, _sigY as sizes in [m] in coordinate representation (_presCA="c") or as angular divergences in [rad] in angular representation (_presCA="a")'],
        ['gbm_ft', 's', 't', 'treat _sigT as pulse duration in [s] in time domain/representation (_presFT="t") or as bandwidth in [eV] in frequency domain/representation (_presFT="f")'],

    #---Calculation Types
        # Electron Trajectory
        ['tr', '', '', 'calculate electron trajectory', 'store_true'],
        ['tr_cti', 'f', 0.0, 'initial time moment (c*t) for electron trajectory calculation [m]'],
        ['tr_ctf', 'f', 0.0, 'final time moment (c*t) for electron trajectory calculation [m]'],
        ['tr_np', 'f', 10000, 'number of points for trajectory calculation'],
        ['tr_mag', 'i', 1, 'magnetic field to be used for trajectory calculation: 1- approximate, 2- accurate'],
        ['tr_fn', 's', 'res_trj.dat', 'file name for saving calculated trajectory data'],
        ['tr_pl', 's', '', 'plot the resulting trajectiry in graph(s): ""- dont plot, otherwise the string should list the trajectory components to plot'],

        #Single-Electron Spectrum vs Photon Energy
        ['ss', '', '', 'calculate single-e spectrum vs photon energy', 'store_true'],
        ['ss_ei', 'f', 100.0, 'initial photon energy [eV] for single-e spectrum vs photon energy calculation'],
        ['ss_ef', 'f', 20000.0, 'final photon energy [eV] for single-e spectrum vs photon energy calculation'],
        ['ss_ne', 'i', 10000, 'number of points vs photon energy for single-e spectrum vs photon energy calculation'],
        ['ss_x', 'f', 0.0, 'horizontal position [m] for single-e spectrum vs photon energy calculation'],
        ['ss_y', 'f', 0.0, 'vertical position [m] for single-e spectrum vs photon energy calculation'],
        ['ss_meth', 'i', 2, 'method to use for single-e spectrum vs photon energy calculation: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"'],
        ['ss_prec', 'f', 0.01, 'relative precision for single-e spectrum vs photon energy calculation (nominal value is 0.01)'],
        ['ss_pol', 'i', 6, 'polarization component to extract after spectrum vs photon energy calculation: 0- Linear Horizontal, 1- Linear Vertical, 2- Linear 45 degrees, 3- Linear 135 degrees, 4- Circular Right, 5- Circular Left, 6- Total'],
        ['ss_mag', 'i', 1, 'magnetic field to be used for single-e spectrum vs photon energy calculation: 1- approximate, 2- accurate'],
        ['ss_ft', 's', 'f', 'presentation/domain: "f"- frequency (photon energy), "t"- time'],
        ['ss_u', 'i', 1, 'electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2), 2- sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on representation (freq. or time)'],
        ['ss_fn', 's', 'res_spec_se.dat', 'file name for saving calculated single-e spectrum vs photon energy'],
        ['ss_pl', 's', '', 'plot the resulting single-e spectrum in a graph: ""- dont plot, "e"- show plot vs photon energy'],

        #Multi-Electron Spectrum vs Photon Energy (taking into account e-beam emittance, energy spread and collection aperture size)
        ['sm', '', '', 'calculate multi-e spectrum vs photon energy', 'store_true'],
        ['sm_ei', 'f', 100.0, 'initial photon energy [eV] for multi-e spectrum vs photon energy calculation'],
        ['sm_ef', 'f', 20000.0, 'final photon energy [eV] for multi-e spectrum vs photon energy calculation'],
        ['sm_ne', 'i', 10000, 'number of points vs photon energy for multi-e spectrum vs photon energy calculation'],
        ['sm_x', 'f', 0.0, 'horizontal center position [m] for multi-e spectrum vs photon energy calculation'],
        ['sm_rx', 'f', 0.001, 'range of horizontal position / horizontal aperture size [m] for multi-e spectrum vs photon energy calculation'],
        ['sm_nx', 'i', 1, 'number of points vs horizontal position for multi-e spectrum vs photon energy calculation'],
        ['sm_y', 'f', 0.0, 'vertical center position [m] for multi-e spectrum vs photon energy calculation'],
        ['sm_ry', 'f', 0.001, 'range of vertical position / vertical aperture size [m] for multi-e spectrum vs photon energy calculation'],
        ['sm_ny', 'i', 1, 'number of points vs vertical position for multi-e spectrum vs photon energy calculation'],
        ['sm_mag', 'i', 1, 'magnetic field to be used for calculation of multi-e spectrum spectrum or intensity distribution: 1- approximate, 2- accurate'],
        ['sm_hi', 'i', 1, 'initial UR spectral harmonic to be taken into account for multi-e spectrum vs photon energy calculation'],
        ['sm_hf', 'i', 15, 'final UR spectral harmonic to be taken into account for multi-e spectrum vs photon energy calculation'],
        ['sm_prl', 'f', 1.0, 'longitudinal integration precision parameter for multi-e spectrum vs photon energy calculation'],
        ['sm_pra', 'f', 1.0, 'azimuthal integration precision parameter for multi-e spectrum vs photon energy calculation'],
        ['sm_meth', 'i', -1, 'method to use for spectrum vs photon energy calculation in case of arbitrary input magnetic field: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler", -1- dont use this accurate integration method (rather use approximate if possible)'],
        ['sm_prec', 'f', 0.01, 'relative precision for spectrum vs photon energy calculation in case of arbitrary input magnetic field (nominal value is 0.01)'],
        ['sm_nm', 'i', 1, 'number of macro-electrons for calculation of spectrum in case of arbitrary input magnetic field'],
        ['sm_na', 'i', 5, 'number of macro-electrons to average on each node at parallel (MPI-based) calculation of spectrum in case of arbitrary input magnetic field'],
        ['sm_ns', 'i', 5, 'saving periodicity (in terms of macro-electrons) for intermediate intensity at calculation of multi-electron spectrum in case of arbitrary input magnetic field'],
        ['sm_type', 'i', 1, 'calculate flux (=1) or flux per unit surface (=2)'],
        ['sm_pol', 'i', 6, 'polarization component to extract after calculation of multi-e flux or intensity: 0- Linear Horizontal, 1- Linear Vertical, 2- Linear 45 degrees, 3- Linear 135 degrees, 4- Circular Right, 5- Circular Left, 6- Total'],
        ['sm_rm', 'i', 1, 'method for generation of pseudo-random numbers for e-beam phase-space integration: 1- standard pseudo-random number generator, 2- Halton sequences, 3- LPtau sequences (to be implemented)'],
        ['sm_fn', 's', 'res_spec_me.dat', 'file name for saving calculated milti-e spectrum vs photon energy'],
        ['sm_pl', 's', '', 'plot the resulting spectrum-e spectrum in a graph: ""- dont plot, "e"- show plot vs photon energy'],
        #to add options for the multi-e calculation from "accurate" magnetic field

        #Power Density Distribution vs horizontal and vertical position
        ['pw', '', '', 'calculate SR power density distribution', 'store_true'],
        ['pw_x', 'f', 0.0, 'central horizontal position [m] for calculation of power density distribution vs horizontal and vertical position'],
        ['pw_rx', 'f', 0.015, 'range of horizontal position [m] for calculation of power density distribution vs horizontal and vertical position'],
        ['pw_nx', 'i', 100, 'number of points vs horizontal position for calculation of power density distribution'],
        ['pw_y', 'f', 0.0, 'central vertical position [m] for calculation of power density distribution vs horizontal and vertical position'],
        ['pw_ry', 'f', 0.015, 'range of vertical position [m] for calculation of power density distribution vs horizontal and vertical position'],
        ['pw_ny', 'i', 100, 'number of points vs vertical position for calculation of power density distribution'],
        ['pw_pr', 'f', 1.0, 'precision factor for calculation of power density distribution'],
        ['pw_meth', 'i', 1, 'power density computation method (1- "near field", 2- "far field")'],
        ['pw_zst', 'f', 0., 'initial longitudinal position along electron trajectory of power density distribution (effective if pow_sst < pow_sfi)'],
        ['pw_zfi', 'f', 0., 'final longitudinal position along electron trajectory of power density distribution (effective if pow_sst < pow_sfi)'],
        ['pw_mag', 'i', 1, 'magnetic field to be used for power density calculation: 1- approximate, 2- accurate'],
        ['pw_fn', 's', 'res_pow.dat', 'file name for saving calculated power density distribution'],
        ['pw_pl', 's', '', 'plot the resulting power density distribution in a graph: ""- dont plot, "x"- vs horizontal position, "y"- vs vertical position, "xy"- vs horizontal and vertical position'],

        #Single-Electron Intensity distribution vs horizontal and vertical position
        ['si', '', '', 'calculate single-e intensity distribution (without wavefront propagation through a beamline) vs horizontal and vertical position', 'store_true'],
        #Single-Electron Wavefront Propagation
        ['ws', '', '', 'calculate single-electron (/ fully coherent) wavefront propagation', 'store_true'],
        #Multi-Electron (partially-coherent) Wavefront Propagation
        ['wm', '', '', 'calculate multi-electron (/ partially coherent) wavefront propagation', 'store_true'],

        ['w_e', 'f', 9000.0, 'photon energy [eV] for calculation of intensity distribution vs horizontal and vertical position'],
        ['w_ef', 'f', -1.0, 'final photon energy [eV] for calculation of intensity distribution vs horizontal and vertical position'],
        ['w_ne', 'i', 1, 'number of points vs photon energy for calculation of intensity distribution'],
        ['w_x', 'f', 0.0, 'central horizontal position [m] for calculation of intensity distribution'],
        ['w_rx', 'f', 0.0004, 'range of horizontal position [m] for calculation of intensity distribution'],
        ['w_nx', 'i', 100, 'number of points vs horizontal position for calculation of intensity distribution'],
        ['w_y', 'f', 0.0, 'central vertical position [m] for calculation of intensity distribution vs horizontal and vertical position'],
        ['w_ry', 'f', 0.0006, 'range of vertical position [m] for calculation of intensity distribution vs horizontal and vertical position'],
        ['w_ny', 'i', 100, 'number of points vs vertical position for calculation of intensity distribution'],
        ['w_smpf', 'f', 1.0, 'sampling factor for calculation of intensity distribution vs horizontal and vertical position'],
        ['w_meth', 'i', 2, 'method to use for calculation of intensity distribution vs horizontal and vertical position: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"'],
        ['w_prec', 'f', 0.01, 'relative precision for calculation of intensity distribution vs horizontal and vertical position'],
        ['w_u', 'i', 1, 'electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2), 2- sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on representation (freq. or time)'],
        ['si_pol', 'i', 6, 'polarization component to extract after calculation of intensity distribution: 0- Linear Horizontal, 1- Linear Vertical, 2- Linear 45 degrees, 3- Linear 135 degrees, 4- Circular Right, 5- Circular Left, 6- Total'],
        ['si_type', 'i', 0, 'type of a characteristic to be extracted after calculation of intensity distribution: 0- Single-Electron Intensity, 1- Multi-Electron Intensity, 2- Single-Electron Flux, 3- Multi-Electron Flux, 4- Single-Electron Radiation Phase, 5- Re(E): Real part of Single-Electron Electric Field, 6- Im(E): Imaginary part of Single-Electron Electric Field, 7- Single-Electron Intensity, integrated over Time or Photon Energy'],
        ['w_mag', 'i', 1, 'magnetic field to be used for calculation of intensity distribution vs horizontal and vertical position: 1- approximate, 2- accurate'],

        ['si_fn', 's', 'res_int_se.dat', 'file name for saving calculated single-e intensity distribution (without wavefront propagation through a beamline) vs horizontal and vertical position'],
        ['si_pl', 's', '', 'plot the input intensity distributions in graph(s): ""- dont plot, "x"- vs horizontal position, "y"- vs vertical position, "xy"- vs horizontal and vertical position'],
        ['ws_fni', 's', ws_fni, 'file name for saving propagated single-e intensity distribution vs horizontal and vertical position'],
        ['ws_pl', 's', '', 'plot the resulting intensity distributions in graph(s): ""- dont plot, "x"- vs horizontal position, "y"- vs vertical position, "xy"- vs horizontal and vertical position'],

        ['wm_nm', 'i', 1000, 'number of macro-electrons (coherent wavefronts) for calculation of multi-electron wavefront propagation'],
        ['wm_na', 'i', 5, 'number of macro-electrons (coherent wavefronts) to average on each node for parallel (MPI-based) calculation of multi-electron wavefront propagation'],
        ['wm_ns', 'i', 5, 'saving periodicity (in terms of macro-electrons / coherent wavefronts) for intermediate intensity at multi-electron wavefront propagation calculation'],
        ['wm_ch', 'i', 0, 'type of a characteristic to be extracted after calculation of multi-electron wavefront propagation: #0- intensity (s0); 1- four Stokes components; 2- mutual intensity cut vs x; 3- mutual intensity cut vs y; 40- intensity(s0), mutual intensity cuts and degree of coherence vs X & Y'],
        ['wm_ap', 'i', 0, 'switch specifying representation of the resulting Stokes parameters: coordinate (0) or angular (1)'],
        ['wm_x0', 'f', 0.0, 'horizontal center position for mutual intensity cut calculation'],
        ['wm_y0', 'f', 0.0, 'vertical center position for mutual intensity cut calculation'],
        ['wm_ei', 'i', 0, 'integration over photon energy is required (1) or not (0); if the integration is required, the limits are taken from w_e, w_ef'],
        ['wm_rm', 'i', 1, 'method for generation of pseudo-random numbers for e-beam phase-space integration: 1- standard pseudo-random number generator, 2- Halton sequences, 3- LPtau sequences (to be implemented)'],
        ['wm_am', 'i', 0, 'multi-electron integration approximation method: 0- no approximation (use the standard 5D integration method), 1- integrate numerically only over e-beam energy spread and use convolution to treat transverse emittance'],
        ['wm_fni', 's', 'res_int_pr_me.dat', 'file name for saving propagated multi-e intensity distribution vs horizontal and vertical position'],
        ['wm_ff', 's', 'ascii', 'format of file name for saving propagated multi-e intensity distribution vs horizontal and vertical position (ascii and hdf5 supported)'],

        ['wm_nmm', 'i', 1, 'number of MPI masters to use'],
        ['wm_ncm', 'i', 100, 'number of Coherent Modes to calculate'],
        ['wm_acm', 's', 'SP', 'coherent mode decomposition algorithm to be used (supported algorithms are: "SP" for SciPy, "SPS" for SciPy Sparse, "PM" for Primme, based on names of software packages)'],
        ['wm_nop', '', '', 'switch forcing to do calculations ignoring any optics defined (by set_optics function)', 'store_true'],

        ['wm_fnmi', 's', '', 'file name of input cross-spectral density / mutual intensity; if this file name is supplied, the initial cross-spectral density (for such operations as coherent mode decomposition) will not be calculated, but rathre it will be taken from that file.'],
        ['wm_fncm', 's', '', 'file name of input coherent modes; if this file name is supplied, the eventual partially-coherent radiation propagation simulation will be done based on propagation of the coherent modes from that file.'],

        ['wm_fbk', '', '', 'create backup file(s) with propagated multi-e intensity distribution vs horizontal and vertical position and other radiation characteristics', 'store_true'],

        # Optics parameters
        ['op_r', 'f', 24.99, 'longitudinal position of the first optical element [m]'],
        # Former appParam:
        ['rs_type', 's', 'g', 'source type, (u) idealized undulator, (t), tabulated undulator, (m) multipole, (g) gaussian beam'],

    #---Beamline optics:
        # zero_drift: drift
        ['op_zero_drift_L', 'f', 0, 'length'],

        # Watchpoint_Toroid: drift
        ['op_Watchpoint_Toroid_L', 'f', 0.010000000000001563, 'length'],

        # Toroid: toroidalMirror
        ['op_Toroid_hfn', 's', '', 'heightProfileFile'],
        ['op_Toroid_dim', 's', 'x', 'orientation'],
        ['op_Toroid_ap_shape', 's', 'r', 'apertureShape'],
        ['op_Toroid_rt', 'f', 7592.12, 'tangentialRadius'],
        ['op_Toroid_rs', 'f', 0.186, 'sagittalRadius'],
        ['op_Toroid_size_tang', 'f', 0.96, 'tangentialSize'],
        ['op_Toroid_size_sag', 'f', 0.08, 'sagittalSize'],
        ['op_Toroid_ang', 'f', 0.007, 'grazingAngle'],
        ['op_Toroid_horizontalPosition', 'f', tm_x, 'horizontalPosition'],
        ['op_Toroid_verticalPosition', 'f', tm_y, 'verticalPosition'],
        ['op_Toroid_nvx', 'f', 0.9999755001000415, 'normalVectorX'],
        ['op_Toroid_nvy', 'f', 0.0, 'normalVectorY'],
        ['op_Toroid_nvz', 'f', -0.006999942833473391, 'normalVectorZ'],
        ['op_Toroid_tvx', 'f', 0.006999942833473391, 'tangentialVectorX'],
        ['op_Toroid_tvy', 'f', 0.0, 'tangentialVectorY'],
        ['op_Toroid_amp_coef', 'f', 1.0, 'heightAmplification'],

        # Toroid_Watchpoint2: drift
        ['op_Toroid_Watchpoint2_L', 'f', 25.0, 'length'],

    #---Propagation parameters
        ['op_zero_drift_pp', 'f',         [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'zero_drift'],
        ['op_Watchpoint_Toroid_pp', 'f',  [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Watchpoint_Toroid'],
        ['op_Toroid_pp', 'f',             [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Toroid'],
        ['op_Toroid_Watchpoint2_pp', 'f', [0, 0, 1.0, 1, 0, 10.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Toroid_Watchpoint2'],
        ['op_fin_pp', 'f',                [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'final post-propagation (resize) parameters'],

        #[ 0]: Auto-Resize (1) or not (0) Before propagation
        #[ 1]: Auto-Resize (1) or not (0) After propagation
        #[ 2]: Relative Precision for propagation with Auto-Resizing (1. is nominal)
        #[ 3]: Allow (1) or not (0) for semi-analytical treatment of the quadratic (leading) phase terms at the propagation
        #[ 4]: Do any Resizing on Fourier side, using FFT, (1) or not (0)
        #[ 5]: Horizontal Range modification factor at Resizing (1. means no modification)
        #[ 6]: Horizontal Resolution modification factor at Resizing
        #[ 7]: Vertical Range modification factor at Resizing
        #[ 8]: Vertical Resolution modification factor at Resizing
        #[ 9]: Type of wavefront Shift before Resizing (not yet implemented)
        #[10]: New Horizontal wavefront Center position after Shift (not yet implemented)
        #[11]: New Vertical wavefront Center position after Shift (not yet implemented)
        #[12]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Horizontal Coordinate
        #[13]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Vertical Coordinate
        #[14]: Optional: Orientation of the Output Optical Axis vector in the Incident Beam Frame: Longitudinal Coordinate
        #[15]: Optional: Orientation of the Horizontal Base vector of the Output Frame in the Incident Beam Frame: Horizontal Coordinate
        #[16]: Optional: Orientation of the Horizontal Base vector of the Output Frame in the Incident Beam Frame: Vertical Coordinate
    ]

    return varParam









