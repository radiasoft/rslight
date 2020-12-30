""" This file pulls in the initialized varParams, set_optics function from sirepo and read_srw_file function from SRW_mp_helper
    You have to manually set which offsets you want to run with (sorry argparse wouldn't work with multiprocessing for some reason :( ) 
    Offsets to set: mirror1 (horizontal) offset ; mirror 2 (vertical) offset
    Rotations : mirror 1                        ; mirror 2                  
    beam outputs will be saved to beam_intensities..._.npy and the offsets for predictions will be saved to parameters_..._.npy .
    You can then run these files in ml_kb_problem_CF.ipynb to predict the offsets given the beam intensities """

import os
import time 
import srwl_bl
import srwlib
import srwlpy
import uti_plot_com as srw_io
import srwl_uti_smp
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process, Lock
import multiprocessing
from SRW_mp_helperARI import * 
import queue 
from queue import Queue
import time 
import argparse
import shutil 

'''
Sirepo SRW beamline source:      https://www.sirepo.com/srw#/beamline/sDw05G14
'''


'''
parser = argparse.ArgumentParser()
parser.add_argument('--n_runs', default=100)
parser.add_argument('--n_processes', default=10)
args = parser.parse_args()
'''

def varparam_idxs(varParam, string):
    """ A function to find the index of the parameters you want to adapt based on the exact string text. Hard-coding led to different idxs depending
    on the version so when adding more parameters I recommend using this function """
    return varParam.index(string)


def run_srw_simulation(task_cut, hOffsetIdx,vOffsetIdx, hOffsetIdx2, vOffsetIdx2, file_idx, nvx_idx, nvy_idx , nvz_idx, nvx2_idx, nvy2_idx , nvz2_idx, wp_idx, process_number):
    """ Runs the number of SRW simulations as determined by the number of rows in task_cut. task_cut holds the parameters that will be updated to varParam to introduce offsets
        task_cut: 
             i: task number 
             offset_notation: binary values for which offsets/rotations are applied:  offsets_mirror1, offsets_mirror2, rotations_mirror1, rotations_mirror2
             dx1: the horizontal offset to mirror 1
             dy1: the vertical offset to mirror 1
             dx2: the horizontal offset to mirror 2 
             dy2: the vertical offset to mirror 2 
             thetax1: thetax for rotation matrix to be applied to mirror 1 
             thetay1: thetay for rotation matrix to be applied to mirror 1 
             thetaz1: thetaz for rotation matrix to be applied to mirror 1
             thetax2: thetax for rotation matrix to be applied to mirror 2
             thetay2: thetay for rotation matrix to be applied to mirror 2 
             thetaz2: thetaz for rotation matrix to be applied to mirror 2 
             wp: watchpoint location for final screen 
        hOffsetIdx: index of where to update horizontal offset in varParams for mirror 1 
        vOffsetIdx: index of where to update vertical offset in varParams for mirror 1 
        hOffsetIdx2: index of where to update horizontal offset in varParams for mirror 2
        vOffsetIdx2: index of where to update vertical offset in varParams for mirror 
        file_idx: the index of the output beam file 
        nvx_idx: index of where to update thetax in varParams for mirror 1 
        nvy_idx: index of where to update thetay in varParams for mirror 1 
        nvz_idx: index of where to update thetaz in varParams for mirror 1 
        nvx2_idx: index of where to update thetax in varParams for mirror 2
        nvy2_idx: index of where to update thetay in varParams for mirror 2 
        nvz2_idx: index of where to update thetaz in varParams for mirror 2 
        wp_idx: index of where to update the final screen 
        process_number: the process number running this task cut
    """
    print('Process ' + str(process_number) + ' to complete ' + str(len(task_cut)) + ' tasks')
    for t in range(len(task_cut)):
        
        varParam = get_reset_varParam()
        
        
        #print(len(task_cut))
        #print(task_cut)

        #### get task parameters based on which combo 
        i, offset_notation, dx1, dy1, dx2, dy2, thetax1, thetay1, thetaz1, thetax2, thetay2, thetaz2, wp = task_cut[t]
        
        offsets_mirror1, offsets_mirror2, rotations_mirror1, rotations_mirror2, watchpoint_pos = offset_notation
        
        task_params = [] 
        #### update params
        if offsets_mirror1:
            varParam[hOffsetIdx][2] = dx1
            task_params.append(dx1.reshape(1,1))
            varParam[vOffsetIdx][2] = dy1
            
            
        if offsets_mirror2:
            varParam[hOffsetIdx2][2] = dx2
            varParam[vOffsetIdx2][2] = dy2
            task_params.append(dy2.reshape(1,1))
        

        if rotations_mirror1: 
            vx = varParam[nvx_idx][2]
            vy = varParam[nvy_idx][2]
            vz = varParam[nvz_idx][2]

            Rx = np.array([[1, 0, 0], [0, np.cos(thetax1), -np.sin(thetax1)], [0, np.sin(thetax1), np.cos(thetax1)]])
            Ry = np.array([[np.cos(thetay1), 0, np.sin(thetay1)], [0, 1, 0], [-np.sin(thetay1), 0, np.cos(thetay1)]])
            Rz = np.array([[np.cos(thetaz1), -np.sin(thetaz1), 0], [np.sin(thetaz1), np.cos(thetaz1), 0], [0, 0, 1]])

            Rxy = np.dot(Rx,Ry)
            R_tot = np.dot(Rxy,Rz)
            v = np.array([vx, vy, vz]).reshape(3,1)

            rtot_v = np.dot(R_tot, v)

            varParam[nvx_idx][2] = rtot_v[0]

            varParam[nvy_idx][2] = rtot_v[1]

            varParam[nvz_idx][2] = rtot_v[2]
            
            #task_params.append(thetax1.reshape(1,1))
            task_params.append(thetay1.reshape(1,1))
            task_params.append(thetaz1.reshape(1,1))
            
        if rotations_mirror2: 
            vx2 = varParam[nvx2_idx][2]
            vy2 = varParam[nvy2_idx][2]
            vz2 = varParam[nvz2_idx][2]

            Rx2 = np.array([[1, 0, 0], [0, np.cos(thetax2), -np.sin(thetax2)], [0, np.sin(thetax2), np.cos(thetax2)]])
            Ry2 = np.array([[np.cos(thetay2), 0, np.sin(thetay2)], [0, 1, 0], [-np.sin(thetay2), 0, np.cos(thetay2)]])
            Rz2 = np.array([[np.cos(thetaz2), -np.sin(thetaz2), 0], [np.sin(thetaz2), np.cos(thetaz2), 0], [0, 0, 1]])

            Rxy2 = np.dot(Rx2,Ry2)
            R_tot2 = np.dot(Rxy2,Rz2)
            v2 = np.array([vx2, vy2, vz2]).reshape(3,1)

            rtot_v2 = np.dot(R_tot2, v2)

            varParam[nvx2_idx][2] = rtot_v2[0]

            varParam[nvy2_idx][2] = rtot_v2[1]

            varParam[nvz2_idx][2] = rtot_v2[2]
            
            task_params.append(thetax2.reshape(1,1))
            #task_params.append(thetay2.reshape(1,1))
            task_params.append(thetaz2.reshape(1,1))
            
        if watchpoint_pos:
            print('updating with ' + str(wp))
            varParam[wp_idx][2] = wp
            task_params.append(wp.reshape(1,1))
            
    
        
        save_dat = 'dat_files/res_int_se_' + str(i) + '.dat'
        varParam[file_idx][2] = save_dat


        v = srwl_bl.srwl_uti_parse_options(varParam, use_sys_argv=True)
        op = set_optics(v)
        v.si = True
        v.si_pl = ''
        v.ws = True
        v.ws_pl = ''
        mag = None
        if v.rs_type == 'm':
            mag = srwlib.SRWLMagFldC()
            mag.arXc.append(0)
            mag.arYc.append(0)
            mag.arMagFld.append(srwlib.SRWLMagFldM(v.mp_field, v.mp_order, v.mp_distribution, v.mp_len))
            mag.arZc.append(v.mp_zc)
        srwl_bl.SRWLBeamline(_name=v.name, _mag_approx=mag).calc_all(v, op)
        beam = read_srw_file(save_dat)
        h = beam.shape[0]
        w = beam.shape[1]
        beam = beam.reshape(1, h, w)
        task_params = np.concatenate(task_params, axis=1)

        np.save('beams/beam_' + str(int(i)) + '.npy', beam)
        np.save('parameters/offsets_' + str(int(i)) + '.npy', task_params)
        #time.sleep(0.5)
        print('Process ' + str(process_number) + ' finished task ' + str(int(i)))
                        

def main():
    ####################### creates temporary storage folders for processes to dump their data 
    if not os.path.exists('beams'):
        os.makedirs('beams')
        
    if not os.path.exists('parameters'):
        os.makedirs('parameters')
        
    if not os.path.exists('dat_files'):
        os.makedirs('dat_files')
        
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
        
    ####################### update this for number of simulations you want to run 
    n_runs = 5000
    
    
    ####################### update this to cpu_count() - 1 if you are running other things on the server at the same time 
    n_processes = multiprocessing.cpu_count() 
    print('Number processes available: ' + str(n_processes))
    if (n_runs < n_processes):
        n_processes = n_runs 
    
    
    ####################### set these ranges for offset/rotation size 
    zx_min = -1e-4
    zx_max = 1e-4
    theta_min = 0
    theta_max = 0.01
    wp_min = 1
    wp_max = 8
    
    ####################### set these values to determine which offsets/rotations you want applied in your dataset 
    offsets_mirror1 = True
    offsets_mirror2 = True
    rotations_mirror1 = True 
    rotations_mirror2 = True 
    watchpoint_pos = True
    
    ####################### uniformly distributed parameter offsets   
    off_dx1_vals = np.random.uniform(zx_min, zx_max, n_runs)
    off_dy1_vals = np.random.uniform(zx_min, zx_max, n_runs)
    off_dx2_vals = np.random.uniform(zx_min, zx_max, n_runs)
    off_dy2_vals = np.random.uniform(zx_min, zx_max, n_runs)
    thetax1s = np.random.uniform(theta_min, theta_max, n_runs)
    thetay1s = np.random.uniform(theta_min, theta_max, n_runs)
    thetaz1s = np.random.uniform(theta_min, theta_max, n_runs)
    thetax2s = np.random.uniform(theta_min, theta_max, n_runs)
    thetay2s = np.random.uniform(theta_min, theta_max, n_runs)
    thetaz2s = np.random.uniform(theta_min, theta_max, n_runs)
    wps = np.random.uniform(wp_min, wp_max, n_runs)
    
    
    ####################### updates where input/output will be saved based on which offsets 
    save_str = ''
    if offsets_mirror1:
        
        save_str += 'mirror1_offsets_'
        
    if offsets_mirror2:
        
        save_str += 'mirror2_offsets_'

    if rotations_mirror1:
        
        save_str += 'mirror1_rotations_'
        
    if rotations_mirror2: 
        
        save_str += 'mirror2_rotations_'
        
    if watchpoint_pos:
        save_str += 'watchpoint_position_'

    
    ####################### gets the index of the parameters of interest within varParam 
    varParam = get_reset_varParam()
    hOffsetIdx = varparam_idxs(varParam, ['op_KBV_x', 'f', 0.0, 'horizontalOffset']) ### mirror 1 horizontal
    vOffsetIdx = varparam_idxs(varParam, ['op_KBV_y', 'f', 0.0, 'verticalOffset']) 
    hOffsetIdx2 = varparam_idxs(varParam, ['op_KBH_x', 'f', 0.0, 'horizontalOffset']) ### mirror 2 horizontal
    vOffsetIdx2 = varparam_idxs(varParam,['op_KBH_y', 'f', 0.0, 'verticalOffset'])  ### mirror 2 vertical 
    
    nvx_idx = varparam_idxs(varParam, ['op_KBV_nvx', 'f', 0.0, 'normalVectorX']) ### mirror 1 normal vector
    nvy_idx = varparam_idxs(varParam, ['op_KBV_nvy', 'f', 0.9990482216385568, 'normalVectorY'])
    nvz_idx = varparam_idxs(varParam, ['op_KBV_nvz', 'f', -0.043619386066714935, 'normalVectorZ'])
    
    nvx2_idx = varparam_idxs(varParam, ['op_KBH_nvx', 'f', 0.9990482216385568, 'normalVectorX']) ### mirror 2 normal vector
    nvy2_idx = varparam_idxs(varParam, ['op_KBH_nvy', 'f', 0.0, 'normalVectorY'])
    nvz2_idx = varparam_idxs(varParam, ['op_KBH_nvz', 'f', -0.043619386066714935, 'normalVectorZ'])
    file_idx = varparam_idxs(varParam,  ['ws_fni', 's', 'res_int_pr_se.dat', 'file name for saving propagated single-e intensity distribution vs horizontal and vertical position'])
    wp_idx = varparam_idxs(varParam, ['op_After_KBH_Sample_L', 'f', 0.12199999999999989, 'length'])
    print('final screen index')
    print(wp_idx)
    start = time.time()
    
    tasks = [] 
    
    ####################### set which offsets/rotations are being applied so this info can be sent to each task 
    offset_notation = [offsets_mirror1, offsets_mirror2, rotations_mirror1, rotations_mirror2, watchpoint_pos] 
    
    ####################### create all the tasks to be run with the offsets and rotations 
    for i in range(n_runs):
        task = np.array([i, offset_notation, off_dx1_vals[i],  off_dy1_vals[i], off_dx2_vals[i],  off_dy2_vals[i], thetax1s[i], thetay1s[i], thetaz1s[i], thetax2s[i], thetay2s[i], thetaz2s[i], wps[i]]).reshape(1, -1)
        tasks.append(task)
    tasks = np.concatenate(tasks)
    
    ####################### splits up the number of tasks to be done evenly across the number of processes available and assigns a task cut to each process 
    processes = [] 
    split = int(n_runs / n_processes)
    idx0 = 0
    idx1 = split

    for j in range(n_processes):
        if j == n_processes - 1:
            idx1 = n_runs
        task_cut = tasks[idx0:idx1]
        p = Process(target=run_srw_simulation, args=(task_cut, hOffsetIdx,vOffsetIdx, hOffsetIdx2, vOffsetIdx2, file_idx, nvx_idx, nvy_idx , nvz_idx, nvx2_idx, nvy2_idx , nvz2_idx, wp_idx, j))
        processes.append(p)
        p.start()
        idx0 += split
        idx1 += split

    print('processes started')    

    
    ####################### must join all processes to end multiprocess thread 
    for p in processes:
        print('about to join ' + str(p))
        p.join()
        print('joined ' + str(p))
    
    end = time.time()
    print('time to run ' + str(n_runs) + ' simulations with ' + str(n_processes) + ' processes: ' + str(np.round((start - end)/60, 4)) + ' minutes')

    
    
    ###### reads in the temporary storage folder info to create one input and one output file for ML 
    beam_arrays = []
    task_arrays = [] 
    for i in range(n_runs):
        beam = np.load('beams/beam_' + str(int(i)) + '.npy')
        beam_arrays.append(beam)
        param = np.load('parameters/offsets_' + str(int(i)) + '.npy')
        task_arrays.append(param)
        
    beams_all = np.concatenate(beam_arrays, axis=0)
    params_all = np.concatenate(task_arrays, axis=0)
    
    
    ###### actual input and output files used for ML 
    np.save('datasets/beam_intensities_' + save_str + str(n_runs) + 'runs.npy', beams_all)
    np.save('datasets/parameters_' + save_str + str(n_runs) + 'runs.npy', params_all)
    
    ####### deletes temporary storage folders for processes to dump their data 
    shutil.rmtree('beams')
    shutil.rmtree('parameters')
    shutil.rmtree('dat_files')

if __name__ == '__main__':
    main()
    
    

        
    