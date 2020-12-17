
import os
import time 
import srwl_bl
import srwlib
import srwlpy
import uti_plot_com as srw_io
import srwl_uti_smp
import numpy as np


def read_srw_file(filename):
    """ This function takes in an srw file and returns the beam data. This was adapted from srwl_uti_dataProcess.py file"""
    data, mode, ranges, labels, units = srw_io.file_load(filename)
    data = np.array(data).reshape((ranges[8], ranges[5]), order='C')
    '''
    return {'data': data,
            'shape': data.shape,
            'mean': np.mean(data),
            'photon_energy': ranges[0],
            'horizontal_extent': ranges[3:5],
            'vertical_extent': ranges[6:8],
            # 'mode': mode,
            'labels': labels,
            'units': units}
    '''
    return data 


def set_optics(v=None):
    """ Function from Sirepo's SRW code, based on https://www.sirepo.com/srw#/beamline/HLLFYnur """
    el = []
    pp = []
    names = ['Fixed_Mask', 'Fixed_Mask_Ap_M1', 'Ap_M1', 'M1', 'M1_M2', 'M2', 'M2_Before_Grating', 'Before_Grating', 'After_Grating', 'After_Grating_Before_H_Slit', 'Before_H_Slit', 'H_Slit', 'H_Slit_Aux_1', 'Aux_1', 'Aux_1_Before_V_Slit', 'Before_V_Slit', 'V_Slit', 'V_Slit_Aux_2', 'Aux_2', 'Aux_2_Before_KBV', 'Before_KBV', 'KBV', 'KBV_KBH', 'KBH', 'After_KBH', 'After_KBH_Sample', 'Sample']
    for el_name in names:
        if el_name == 'Fixed_Mask':
            # Fixed_Mask: aperture 18.75m
            el.append(srwlib.SRWLOptA(
                _shape=v.op_Fixed_Mask_shape,
                _ap_or_ob='a',
                _Dx=v.op_Fixed_Mask_Dx,
                _Dy=v.op_Fixed_Mask_Dy,
                _x=v.op_Fixed_Mask_x,
                _y=v.op_Fixed_Mask_y,
            ))
            pp.append(v.op_Fixed_Mask_pp)
        elif el_name == 'Fixed_Mask_Ap_M1':
            # Fixed_Mask_Ap_M1: drift 18.75m
            el.append(srwlib.SRWLOptD(
                _L=v.op_Fixed_Mask_Ap_M1_L,
            ))
            pp.append(v.op_Fixed_Mask_Ap_M1_pp)
        elif el_name == 'Ap_M1':
            # Ap_M1: aperture 27.25m
            el.append(srwlib.SRWLOptA(
                _shape=v.op_Ap_M1_shape,
                _ap_or_ob='a',
                _Dx=v.op_Ap_M1_Dx,
                _Dy=v.op_Ap_M1_Dy,
                _x=v.op_Ap_M1_x,
                _y=v.op_Ap_M1_y,
            ))
            pp.append(v.op_Ap_M1_pp)
        elif el_name == 'M1':
            # M1: ellipsoidMirror 27.25m
            el.append(srwlib.SRWLOptMirEl(
                _p=v.op_M1_p,
                _q=v.op_M1_q,
                _ang_graz=v.op_M1_ang,
                _size_tang=v.op_M1_size_tang,
                _size_sag=v.op_M1_size_sag,
                _nvx=v.op_M1_nvx,
                _nvy=v.op_M1_nvy,
                _nvz=v.op_M1_nvz,
                _tvx=v.op_M1_tvx,
                _tvy=v.op_M1_tvy,
                _x=v.op_M1_x,
                _y=v.op_M1_y,
            ))
            pp.append(v.op_M1_pp)
            mirror_file = v.op_M1_hfn
            assert os.path.isfile(mirror_file), \
                'Missing input file {}, required by M1 beamline element'.format(mirror_file)
            el.append(srwlib.srwl_opt_setup_surf_height_2d(
                srwlib.srwl_uti_read_data_cols(mirror_file, "\t"),
                _dim=v.op_M1_dim,
                _ang=abs(v.op_M1_ang),
                _amp_coef=v.op_M1_amp_coef,
            ))
            pp.append([0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0])
        elif el_name == 'M1_M2':
            # M1_M2: drift 27.25m
            el.append(srwlib.SRWLOptD(
                _L=v.op_M1_M2_L,
            ))
            pp.append(v.op_M1_M2_pp)
        elif el_name == 'M2':
            # M2: mirror 29.35m
            mirror_file = v.op_M2_hfn
            assert os.path.isfile(mirror_file), \
                'Missing input file {}, required by M2 beamline element'.format(mirror_file)
            el.append(srwlib.srwl_opt_setup_surf_height_2d(
                srwlib.srwl_uti_read_data_cols(mirror_file, "\t"),
                _dim=v.op_M2_dim,
                _ang=abs(v.op_M2_ang),
                _amp_coef=v.op_M2_amp_coef,
                _size_x=v.op_M2_size_x,
                _size_y=v.op_M2_size_y,
            ))
            pp.append(v.op_M2_pp)
        elif el_name == 'M2_Before_Grating':
            # M2_Before_Grating: drift 29.35m
            el.append(srwlib.SRWLOptD(
                _L=v.op_M2_Before_Grating_L,
            ))
            pp.append(v.op_M2_Before_Grating_pp)
        elif el_name == 'Before_Grating':
            # Before_Grating: watch 29.75m
            pass
        elif el_name == 'After_Grating':
            # After_Grating: watch 29.75m
            pass
        elif el_name == 'After_Grating_Before_H_Slit':
            # After_Grating_Before_H_Slit: drift 29.75m
            el.append(srwlib.SRWLOptD(
                _L=v.op_After_Grating_Before_H_Slit_L,
            ))
            pp.append(v.op_After_Grating_Before_H_Slit_pp)
        elif el_name == 'Before_H_Slit':
            # Before_H_Slit: watch 36.25m
            pass
        elif el_name == 'H_Slit':
            # H_Slit: aperture 36.25m
            el.append(srwlib.SRWLOptA(
                _shape=v.op_H_Slit_shape,
                _ap_or_ob='a',
                _Dx=v.op_H_Slit_Dx,
                _Dy=v.op_H_Slit_Dy,
                _x=v.op_H_Slit_x,
                _y=v.op_H_Slit_y,
            ))
            pp.append(v.op_H_Slit_pp)
        elif el_name == 'H_Slit_Aux_1':
            # H_Slit_Aux_1: drift 36.25m
            el.append(srwlib.SRWLOptD(
                _L=v.op_H_Slit_Aux_1_L,
            ))
            pp.append(v.op_H_Slit_Aux_1_pp)
        elif el_name == 'Aux_1':
            # Aux_1: watch 36.5m
            pass
        elif el_name == 'Aux_1_Before_V_Slit':
            # Aux_1_Before_V_Slit: drift 36.5m
            el.append(srwlib.SRWLOptD(
                _L=v.op_Aux_1_Before_V_Slit_L,
            ))
            pp.append(v.op_Aux_1_Before_V_Slit_pp)
        elif el_name == 'Before_V_Slit':
            # Before_V_Slit: watch 41.25m
            pass
        elif el_name == 'V_Slit':
            # V_Slit: aperture 41.25m
            el.append(srwlib.SRWLOptA(
                _shape=v.op_V_Slit_shape,
                _ap_or_ob='a',
                _Dx=v.op_V_Slit_Dx,
                _Dy=v.op_V_Slit_Dy,
                _x=v.op_V_Slit_x,
                _y=v.op_V_Slit_y,
            ))
            pp.append(v.op_V_Slit_pp)
        elif el_name == 'V_Slit_Aux_2':
            # V_Slit_Aux_2: drift 41.25m
            el.append(srwlib.SRWLOptD(
                _L=v.op_V_Slit_Aux_2_L,
            ))
            pp.append(v.op_V_Slit_Aux_2_pp)
        elif el_name == 'Aux_2':
            # Aux_2: watch 41.35m
            pass
        elif el_name == 'Aux_2_Before_KBV':
            # Aux_2_Before_KBV: drift 41.35m
            el.append(srwlib.SRWLOptD(
                _L=v.op_Aux_2_Before_KBV_L,
            ))
            pp.append(v.op_Aux_2_Before_KBV_pp)
        elif el_name == 'Before_KBV':
            # Before_KBV: watch 69.334m
            pass
        elif el_name == 'KBV':
            # KBV: ellipsoidMirror 69.334m
            el.append(srwlib.SRWLOptMirEl(
                _p=v.op_KBV_p,
                _q=v.op_KBV_q,
                _ang_graz=v.op_KBV_ang,
                _size_tang=v.op_KBV_size_tang,
                _size_sag=v.op_KBV_size_sag,
                _nvx=v.op_KBV_nvx,
                _nvy=v.op_KBV_nvy,
                _nvz=v.op_KBV_nvz,
                _tvx=v.op_KBV_tvx,
                _tvy=v.op_KBV_tvy,
                _x=v.op_KBV_x,
                _y=v.op_KBV_y,
            ))
            pp.append(v.op_KBV_pp)
            mirror_file = v.op_KBV_hfn
            assert os.path.isfile(mirror_file), \
                'Missing input file {}, required by KBV beamline element'.format(mirror_file)
            el.append(srwlib.srwl_opt_setup_surf_height_2d(
                srwlib.srwl_uti_read_data_cols(mirror_file, "\t"),
                _dim=v.op_KBV_dim,
                _ang=abs(v.op_KBV_ang),
                _amp_coef=v.op_KBV_amp_coef,
            ))
            pp.append([0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0])
        elif el_name == 'KBV_KBH':
            # KBV_KBH: drift 69.334m
            el.append(srwlib.SRWLOptD(
                _L=v.op_KBV_KBH_L,
            ))
            pp.append(v.op_KBV_KBH_pp)
        elif el_name == 'KBH':
            # KBH: ellipsoidMirror 69.628m
            el.append(srwlib.SRWLOptMirEl(
                _p=v.op_KBH_p,
                _q=v.op_KBH_q,
                _ang_graz=v.op_KBH_ang,
                _size_tang=v.op_KBH_size_tang,
                _size_sag=v.op_KBH_size_sag,
                _nvx=v.op_KBH_nvx,
                _nvy=v.op_KBH_nvy,
                _nvz=v.op_KBH_nvz,
                _tvx=v.op_KBH_tvx,
                _tvy=v.op_KBH_tvy,
                _x=v.op_KBH_x,
                _y=v.op_KBH_y,
            ))
            pp.append(v.op_KBH_pp)
            mirror_file = v.op_KBH_hfn
            assert os.path.isfile(mirror_file), \
                'Missing input file {}, required by KBH beamline element'.format(mirror_file)
            el.append(srwlib.srwl_opt_setup_surf_height_2d(
                srwlib.srwl_uti_read_data_cols(mirror_file, "\t"),
                _dim=v.op_KBH_dim,
                _ang=abs(v.op_KBH_ang),
                _amp_coef=v.op_KBH_amp_coef,
            ))
            pp.append([0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0])
        elif el_name == 'After_KBH':
            # After_KBH: watch 69.628m
            pass
        elif el_name == 'After_KBH_Sample':
            # After_KBH_Sample: drift 69.628m
            el.append(srwlib.SRWLOptD(
                _L=v.op_After_KBH_Sample_L,
            ))
            pp.append(v.op_After_KBH_Sample_pp)
        elif el_name == 'Sample':
            # Sample: watch 69.75m
            pass
    pp.append(v.op_fin_pp)
    return srwlib.SRWLOptC(el, pp)




def get_reset_varParam():
    """ Code from Sirepo's SRW code, adapted to a function to give each process their own copy of the default parameters to run """
    varParam = srwl_bl.srwl_uti_ext_options([
    ['name', 's', 'RIXS-ARI 250eV 02 2', 'simulation name'],

#---Data Folder
    ['fdir', 's', '', 'folder (directory) name for reading-in input and saving output data files'],

#---Electron Beam
    ['ebm_nm', 's', '', 'standard electron beam name'],
    ['ebm_nms', 's', '', 'standard electron beam name suffix: e.g. can be Day1, Final'],
    ['ebm_i', 'f', 0.5, 'electron beam current [A]'],
    ['ebm_e', 'f', 3.0, 'electron beam avarage energy [GeV]'],
    ['ebm_de', 'f', 0.0, 'electron beam average energy deviation [GeV]'],
    ['ebm_x', 'f', 0.0, 'electron beam initial average horizontal position [m]'],
    ['ebm_y', 'f', 0.0, 'electron beam initial average vertical position [m]'],
    ['ebm_xp', 'f', 0.0, 'electron beam initial average horizontal angle [rad]'],
    ['ebm_yp', 'f', 0.0, 'electron beam initial average vertical angle [rad]'],
    ['ebm_z', 'f', 0., 'electron beam initial average longitudinal position [m]'],
    ['ebm_dr', 'f', -0.001, 'electron beam longitudinal drift [m] to be performed before a required calculation'],
    ['ebm_ens', 'f', 0.00089, 'electron beam relative energy spread'],
    ['ebm_emx', 'f', 7.6e-10, 'electron beam horizontal emittance [m]'],
    ['ebm_emy', 'f', 8e-12, 'electron beam vertical emittance [m]'],
    # Definition of the beam through Twiss:
    ['ebm_betax', 'f', 1.84, 'horizontal beta-function [m]'],
    ['ebm_betay', 'f', 1.17, 'vertical beta-function [m]'],
    ['ebm_alphax', 'f', 0.0, 'horizontal alpha-function [rad]'],
    ['ebm_alphay', 'f', 0.0, 'vertical alpha-function [rad]'],
    ['ebm_etax', 'f', 0.0, 'horizontal dispersion function [m]'],
    ['ebm_etay', 'f', 0.0, 'vertical dispersion function [m]'],
    ['ebm_etaxp', 'f', 0.0, 'horizontal dispersion function derivative [rad]'],
    ['ebm_etayp', 'f', 0.0, 'vertical dispersion function derivative [rad]'],

#---Undulator
    ['und_bx', 'f', 0.0, 'undulator horizontal peak magnetic field [T]'],
    ['und_by', 'f', 0.425, 'undulator vertical peak magnetic field [T]'],
    ['und_phx', 'f', 0.0, 'initial phase of the horizontal magnetic field [rad]'],
    ['und_phy', 'f', 0.0, 'initial phase of the vertical magnetic field [rad]'],
    ['und_b2e', '', '', 'estimate undulator fundamental photon energy (in [eV]) for the amplitude of sinusoidal magnetic field defined by und_b or und_bx, und_by', 'store_true'],
    ['und_e2b', '', '', 'estimate undulator field amplitude (in [T]) for the photon energy defined by w_e', 'store_true'],
    ['und_per', 'f', 0.07, 'undulator period [m]'],
    ['und_len', 'f', 2.0, 'undulator length [m]'],
    ['und_zc', 'f', -1.25, 'undulator center longitudinal position [m]'],
    ['und_sx', 'i', -1, 'undulator horizontal magnetic field symmetry vs longitudinal position'],
    ['und_sy', 'i', 1, 'undulator vertical magnetic field symmetry vs longitudinal position'],
    ['und_g', 'f', 6.72, 'undulator gap [mm] (assumes availability of magnetic measurement or simulation data)'],
    ['und_ph', 'f', 0.0, 'shift of magnet arrays [mm] for which the field should be set up'],
    ['und_mdir', 's', '', 'name of magnetic measurements sub-folder'],
    ['und_mfs', 's', '', 'name of magnetic measurements for different gaps summary file'],



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
    ['ss_ei', 'f', 10.0, 'initial photon energy [eV] for single-e spectrum vs photon energy calculation'],
    ['ss_ef', 'f', 2100.0, 'final photon energy [eV] for single-e spectrum vs photon energy calculation'],
    ['ss_ne', 'i', 2000, 'number of points vs photon energy for single-e spectrum vs photon energy calculation'],
    ['ss_x', 'f', 0.0, 'horizontal position [m] for single-e spectrum vs photon energy calculation'],
    ['ss_y', 'f', 0.0, 'vertical position [m] for single-e spectrum vs photon energy calculation'],
    ['ss_meth', 'i', 1, 'method to use for single-e spectrum vs photon energy calculation: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"'],
    ['ss_prec', 'f', 0.01, 'relative precision for single-e spectrum vs photon energy calculation (nominal value is 0.01)'],
    ['ss_pol', 'i', 6, 'polarization component to extract after spectrum vs photon energy calculation: 0- Linear Horizontal, 1- Linear Vertical, 2- Linear 45 degrees, 3- Linear 135 degrees, 4- Circular Right, 5- Circular Left, 6- Total'],
    ['ss_mag', 'i', 1, 'magnetic field to be used for single-e spectrum vs photon energy calculation: 1- approximate, 2- accurate'],
    ['ss_ft', 's', 'f', 'presentation/domain: "f"- frequency (photon energy), "t"- time'],
    ['ss_u', 'i', 1, 'electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2), 2- sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on representation (freq. or time)'],
    ['ss_fn', 's', 'res_spec_se.dat', 'file name for saving calculated single-e spectrum vs photon energy'],
    ['ss_pl', 's', '', 'plot the resulting single-e spectrum in a graph: ""- dont plot, "e"- show plot vs photon energy'],

    #Multi-Electron Spectrum vs Photon Energy (taking into account e-beam emittance, energy spread and collection aperture size)
    ['sm', '', '', 'calculate multi-e spectrum vs photon energy', 'store_true'],
    ['sm_ei', 'f', 10.0, 'initial photon energy [eV] for multi-e spectrum vs photon energy calculation'],
    ['sm_ef', 'f', 1000.0, 'final photon energy [eV] for multi-e spectrum vs photon energy calculation'],
    ['sm_ne', 'i', 5000, 'number of points vs photon energy for multi-e spectrum vs photon energy calculation'],
    ['sm_x', 'f', 0.0, 'horizontal center position [m] for multi-e spectrum vs photon energy calculation'],
    ['sm_rx', 'f', 0.008, 'range of horizontal position / horizontal aperture size [m] for multi-e spectrum vs photon energy calculation'],
    ['sm_nx', 'i', 1, 'number of points vs horizontal position for multi-e spectrum vs photon energy calculation'],
    ['sm_y', 'f', 0.0, 'vertical center position [m] for multi-e spectrum vs photon energy calculation'],
    ['sm_ry', 'f', 0.008, 'range of vertical position / vertical aperture size [m] for multi-e spectrum vs photon energy calculation'],
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
    ['pw_rx', 'f', 0.05, 'range of horizontal position [m] for calculation of power density distribution vs horizontal and vertical position'],
    ['pw_nx', 'i', 100, 'number of points vs horizontal position for calculation of power density distribution'],
    ['pw_y', 'f', 0.0, 'central vertical position [m] for calculation of power density distribution vs horizontal and vertical position'],
    ['pw_ry', 'f', 0.05, 'range of vertical position [m] for calculation of power density distribution vs horizontal and vertical position'],
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

    ['w_e', 'f', 250.0, 'photon energy [eV] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ef', 'f', -1.0, 'final photon energy [eV] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ne', 'i', 1, 'number of points vs photon energy for calculation of intensity distribution'],
    ['w_x', 'f', 0.0, 'central horizontal position [m] for calculation of intensity distribution'],
    ['w_rx', 'f', 0.006, 'range of horizontal position [m] for calculation of intensity distribution'],
    ['w_nx', 'i', 100, 'number of points vs horizontal position for calculation of intensity distribution'],
    ['w_y', 'f', 0.0, 'central vertical position [m] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ry', 'f', 0.006, 'range of vertical position [m] for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_ny', 'i', 100, 'number of points vs vertical position for calculation of intensity distribution'],
    ['w_smpf', 'f', 0.3, 'sampling factor for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_meth', 'i', 1, 'method to use for calculation of intensity distribution vs horizontal and vertical position: 0- "manual", 1- "auto-undulator", 2- "auto-wiggler"'],
    ['w_prec', 'f', 0.01, 'relative precision for calculation of intensity distribution vs horizontal and vertical position'],
    ['w_u', 'i', 1, 'electric field units: 0- arbitrary, 1- sqrt(Phot/s/0.1%bw/mm^2), 2- sqrt(J/eV/mm^2) or sqrt(W/mm^2), depending on representation (freq. or time)'],
    ['si_pol', 'i', 6, 'polarization component to extract after calculation of intensity distribution: 0- Linear Horizontal, 1- Linear Vertical, 2- Linear 45 degrees, 3- Linear 135 degrees, 4- Circular Right, 5- Circular Left, 6- Total'],
    ['si_type', 'i', 0, 'type of a characteristic to be extracted after calculation of intensity distribution: 0- Single-Electron Intensity, 1- Multi-Electron Intensity, 2- Single-Electron Flux, 3- Multi-Electron Flux, 4- Single-Electron Radiation Phase, 5- Re(E): Real part of Single-Electron Electric Field, 6- Im(E): Imaginary part of Single-Electron Electric Field, 7- Single-Electron Intensity, integrated over Time or Photon Energy'],
    ['w_mag', 'i', 1, 'magnetic field to be used for calculation of intensity distribution vs horizontal and vertical position: 1- approximate, 2- accurate'],

    ['si_fn', 's', 'res_int_se.dat', 'file name for saving calculated single-e intensity distribution (without wavefront propagation through a beamline) vs horizontal and vertical position'],
    ['si_pl', 's', '', 'plot the input intensity distributions in graph(s): ""- dont plot, "x"- vs horizontal position, "y"- vs vertical position, "xy"- vs horizontal and vertical position'],
    ['ws_fni', 's', 'res_int_pr_se.dat', 'file name for saving propagated single-e intensity distribution vs horizontal and vertical position'],
    ['ws_pl', 's', '', 'plot the resulting intensity distributions in graph(s): ""- dont plot, "x"- vs horizontal position, "y"- vs vertical position, "xy"- vs horizontal and vertical position'],

    ['wm_nm', 'i', 100000, 'number of macro-electrons (coherent wavefronts) for calculation of multi-electron wavefront propagation'],
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
    ['wm_fbk', '', '', 'create backup file(s) with propagated multi-e intensity distribution vs horizontal and vertical position and other radiation characteristics', 'store_true'],

    #to add options
    ['op_r', 'f', 18.75, 'longitudinal position of the first optical element [m]'],
    # Former appParam:
    ['rs_type', 's', 'u', 'source type, (u) idealized undulator, (t), tabulated undulator, (m) multipole, (g) gaussian beam'],

#---Beamline optics:
    # Fixed_Mask: aperture
    ['op_Fixed_Mask_shape', 's', 'r', 'shape'],
    ['op_Fixed_Mask_Dx', 'f', 0.006, 'horizontalSize'],
    ['op_Fixed_Mask_Dy', 'f', 0.006, 'verticalSize'],
    ['op_Fixed_Mask_x', 'f', 0.0, 'horizontalOffset'],
    ['op_Fixed_Mask_y', 'f', 0.0, 'verticalOffset'],

    # Fixed_Mask_Ap_M1: drift
    ['op_Fixed_Mask_Ap_M1_L', 'f', 8.5, 'length'],

    # Ap_M1: aperture
    ['op_Ap_M1_shape', 's', 'r', 'shape'],
    ['op_Ap_M1_Dx', 'f', 0.0088, 'horizontalSize'],
    ['op_Ap_M1_Dy', 'f', 0.01, 'verticalSize'],
    ['op_Ap_M1_x', 'f', 0.0, 'horizontalOffset'],
    ['op_Ap_M1_y', 'f', 0.0, 'verticalOffset'],

    # M1: ellipsoidMirror
    ['op_M1_hfn', 's', 'OlegM1_200tan2000sagFractBeta3.txt', 'heightProfileFile'],
    ['op_M1_dim', 's', 'x', 'orientation'],
    ['op_M1_p', 'f', 28.5, 'firstFocusLength'],
    ['op_M1_q', 'f', 9.0, 'focalLength'],
    ['op_M1_ang', 'f', 0.03490658503988659, 'grazingAngle'],
    ['op_M1_amp_coef', 'f', 1.0, 'heightAmplification'],
    ['op_M1_size_tang', 'f', 0.25, 'tangentialSize'],
    ['op_M1_size_sag', 'f', 0.01, 'sagittalSize'],
    ['op_M1_nvx', 'f', 0.9993908270190958, 'normalVectorX'],
    ['op_M1_nvy', 'f', 0.0, 'normalVectorY'],
    ['op_M1_nvz', 'f', -0.03489949670250097, 'normalVectorZ'],
    ['op_M1_tvx', 'f', 0.03489949670250097, 'tangentialVectorX'],
    ['op_M1_tvy', 'f', 0.0, 'tangentialVectorY'],
    ['op_M1_x', 'f', 0.0, 'horizontalOffset'],
    ['op_M1_y', 'f', 0.0, 'verticalOffset'],

    # M1_M2: drift
    ['op_M1_M2_L', 'f', 2.1000000000000014, 'length'],

    # M2: mirror
    ['op_M2_hfn', 's', 'OlegM2_200tan500sagFractBeta3.txt', 'heightProfileFile'],
    ['op_M2_dim', 's', 'y', 'orientation'],
    ['op_M2_ang', 'f', 0.0440043, 'grazingAngle'],
    ['op_M2_amp_coef', 'f', 1.0, 'heightAmplification'],
    ['op_M2_size_x', 'f', 0.4, 'horizontalTransverseSize'],
    ['op_M2_size_y', 'f', 0.012, 'verticalTransverseSize'],

    # M2_Before_Grating: drift
    ['op_M2_Before_Grating_L', 'f', 0.3999999999999986, 'length'],

    # After_Grating_Before_H_Slit: drift
    ['op_After_Grating_Before_H_Slit_L', 'f', 6.5, 'length'],

    # H_Slit: aperture
    ['op_H_Slit_shape', 's', 'r', 'shape'],
    ['op_H_Slit_Dx', 'f', 5e-05, 'horizontalSize'],
    ['op_H_Slit_Dy', 'f', 0.02, 'verticalSize'],
    ['op_H_Slit_x', 'f', 0.0, 'horizontalOffset'],
    ['op_H_Slit_y', 'f', 0.0, 'verticalOffset'],

    # H_Slit_Aux_1: drift
    ['op_H_Slit_Aux_1_L', 'f', 0.25, 'length'],

    # Aux_1_Before_V_Slit: drift
    ['op_Aux_1_Before_V_Slit_L', 'f', 4.75, 'length'],

    # V_Slit: aperture
    ['op_V_Slit_shape', 's', 'r', 'shape'],
    ['op_V_Slit_Dx', 'f', 0.01, 'horizontalSize'],
    ['op_V_Slit_Dy', 'f', 2e-05, 'verticalSize'],
    ['op_V_Slit_x', 'f', 0.0, 'horizontalOffset'],
    ['op_V_Slit_y', 'f', 0.0, 'verticalOffset'],

    # V_Slit_Aux_2: drift
    ['op_V_Slit_Aux_2_L', 'f', 0.10000000000000142, 'length'],

    # Aux_2_Before_KBV: drift
    ['op_Aux_2_Before_KBV_L', 'f', 27.984, 'length'],

    # KBV: ellipsoidMirror
    ['op_KBV_hfn', 's', 'OlegKBVrixs_200tan500sagFractBeta3.txt', 'heightProfileFile'],
    ['op_KBV_dim', 's', 'y', 'orientation'],
    ['op_KBV_p', 'f', 28.084, 'firstFocusLength'],
    ['op_KBV_q', 'f', 0.416, 'focalLength'],
    ['op_KBV_ang', 'f', 0.043633229999999995, 'grazingAngle'],
    ['op_KBV_amp_coef', 'f', 1.0, 'heightAmplification'],
    ['op_KBV_size_tang', 'f', 0.377, 'tangentialSize'],
    ['op_KBV_size_sag', 'f', 0.03, 'sagittalSize'],
    ['op_KBV_nvx', 'f', 0.0, 'normalVectorX'],
    ['op_KBV_nvy', 'f', 0.9990482216385568, 'normalVectorY'],
    ['op_KBV_nvz', 'f', -0.043619386066714935, 'normalVectorZ'],
    ['op_KBV_tvx', 'f', 0.0, 'tangentialVectorX'],
    ['op_KBV_tvy', 'f', 0.043619386066714935, 'tangentialVectorY'],
    ['op_KBV_x', 'f', 0.0, 'horizontalOffset'],
    ['op_KBV_y', 'f', 0.0, 'verticalOffset'],

    # KBV_KBH: drift
    ['op_KBV_KBH_L', 'f', 0.29399999999999693, 'length'],

    # KBH: ellipsoidMirror
    ['op_KBH_hfn', 's', 'OlegKBHrixs_600tan1000sagFractBeta3.txt', 'heightProfileFile'],
    ['op_KBH_dim', 's', 'x', 'orientation'],
    ['op_KBH_p', 'f', 33.378, 'firstFocusLength'],
    ['op_KBH_q', 'f', 0.122, 'focalLength'],
    ['op_KBH_ang', 'f', 0.043633229999999995, 'grazingAngle'],
    ['op_KBH_amp_coef', 'f', 1.0, 'heightAmplification'],
    ['op_KBH_size_tang', 'f', 0.171, 'tangentialSize'],
    ['op_KBH_size_sag', 'f', 0.02, 'sagittalSize'],
    ['op_KBH_nvx', 'f', 0.9990482216385568, 'normalVectorX'],
    ['op_KBH_nvy', 'f', 0.0, 'normalVectorY'],
    ['op_KBH_nvz', 'f', -0.043619386066714935, 'normalVectorZ'],
    ['op_KBH_tvx', 'f', 0.043619386066714935, 'tangentialVectorX'],
    ['op_KBH_tvy', 'f', 0.0, 'tangentialVectorY'],
    ['op_KBH_x', 'f', 0.0, 'horizontalOffset'],
    ['op_KBH_y', 'f', 0.0, 'verticalOffset'],

    # After_KBH_Sample: drift
    ['op_After_KBH_Sample_L', 'f', 0.12199999999999989, 'length'],

#---Propagation parameters
    ['op_Fixed_Mask_pp', 'f',                  [0, 0, 1.0, 0, 0, 1.2, 8.0, 2.5, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Fixed_Mask'],
    ['op_Fixed_Mask_Ap_M1_pp', 'f',            [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Fixed_Mask_Ap_M1'],
    ['op_Ap_M1_pp', 'f',                       [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Ap_M1'],
    ['op_M1_pp', 'f',                          [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'M1'],
    ['op_M1_M2_pp', 'f',                       [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'M1_M2'],
    ['op_M2_pp', 'f',                          [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'M2'],
    ['op_M2_Before_Grating_pp', 'f',           [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'M2_Before_Grating'],
    ['op_After_Grating_Before_H_Slit_pp', 'f', [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'After_Grating_Before_H_Slit'],
    ['op_H_Slit_pp', 'f',                      [0, 0, 1.0, 0, 0, 0.3, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'H_Slit'],
    ['op_H_Slit_Aux_1_pp', 'f',                [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'H_Slit_Aux_1'],
    ['op_Aux_1_Before_V_Slit_pp', 'f',         [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Aux_1_Before_V_Slit'],
    ['op_V_Slit_pp', 'f',                      [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'V_Slit'],
    ['op_V_Slit_Aux_2_pp', 'f',                [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'V_Slit_Aux_2'],
    ['op_Aux_2_Before_KBV_pp', 'f',            [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'Aux_2_Before_KBV'],
    ['op_KBV_pp', 'f',                         [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'KBV'],
    ['op_KBV_KBH_pp', 'f',                     [0, 0, 1.0, 1, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'KBV_KBH'],
    ['op_KBH_pp', 'f',                         [0, 0, 1.0, 0, 0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'KBH'],
    ['op_After_KBH_Sample_pp', 'f',            [0, 0, 1.0, 4, 0, 0.5, 1.0, 0.5, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'After_KBH_Sample'],
    ['op_fin_pp', 'f',                         [0, 0, 1.0, 0, 0, 1.0, 1.0, 0.2, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'final post-propagation (resize) parameters'],

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
])

    return varParam
