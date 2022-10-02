# -*- coding: utf-8 -*-
"""
Created on Thu May  5 18:43:12 2022

@author: ardit
"""

"""
=== USAGE ===
A path is required when launching the script. This must be the path to a folder
which contains a subfolder named "data" with all needed .dat files. This script
will create a subfolder named "analysis" in the path (at the same level of "data"
subfolder) where output files will be stored.
The path must be relative to this script.
I.E.:
    Folder structure:
        - analysis.py
        - Measures
            > 00
            > 01
                * data
                    # *.dat
            > 02
    Relative path:
        Measures/01/
"""
import time
import os
from functions.Pedestal import pedestal
from functions.Waveform import waveform
from functions.Transfer import transfer
from functions.Threshold import threshold
from functions.ENC import enc
from functions.SelfTrigger import selfTrigger
# from functions.ENC import enc
#%% Preliminary setup
# visual check of current working dir
# os.chdir('C:/Users/xxxxx/xxxxxx/xxxxxx/Tesi');<--set manually dir
os.chdir('C:/Users/ardit/Desktop/Universita/Tesi');
print(os.getcwd());


#%% Global variables

nCh = 32 # number of channels
nTau = 8  # number of tau (peaking times)
n_fthr = 8  # number of fine thrimming settings


plotPedestal = False
plotWaveform = False
plotTransfer = False
plotThreshold = False
plotENC= False
plotSelfTrigger= True

only = 'selfTrigger'

#%% Getting where data are stored
print("Insert path to ASIC number folder (relative):")
sep = os.path.sep
path =str(os.path.abspath(input())) 
if (not(os.path.isdir(path))):
    raise OSError("File not found")
    exit()
    
#%% Data should be under a folder called 'data'
start=time.time()#<--- start time

input_path = path + sep + 'data' + sep
if (not(path.endswith(sep))):
    path = path + sep
# Analysis outputs must be under a folder called 'analysis'
out_path = path + 'analysis' + sep
if (not(os.path.exists(out_path))):
    os.makedirs(out_path)
# Visual check
print(path, input_path, out_path)

#%% Pedestals computation
""" === PEDESTAL ==="""
if (only == '' or only == 'pedestal' or only == 'enc'):
    out_path_current = out_path + 'Pedestal' + sep
    if (not(os.path.exists(out_path_current))):
        os.makedirs(out_path_current)

    fitted_std=pedestal(input_path, out_path_current,nTau,nCh,plotPedestal)
    
""" === WAVEFORM === """
if (only == '' or only == 'waveform'):
    out_path_current = out_path + 'Waveform' + sep
    if (not(os.path.exists(out_path_current))):
        os.makedirs(out_path_current)
        
    waveform(input_path, out_path_current,nTau,nCh,plotWaveform)

"""=== TRANSFER FUNCTION ===  """
if (only == '' or only == 'transfer' or only == 'enc'):
    out_path_current = out_path + 'Transfer' + sep    
    if (not(os.path.exists(out_path_current))):
        os.makedirs(out_path_current)

    linear_fit_coeff,cubic_fit_coeff=transfer(input_path, out_path_current, nCh, nTau, plotTransfer) 

"""=== THRESHOLD SCAN ===  """
if (only == '' or only == 'threshold'):
    out_path_current = out_path + 'Threshold' + sep
    if (not(os.path.exists(out_path_current))):
        os.makedirs(out_path_current)
        
    threshold(input_path, out_path_current,nTau,nCh,n_fthr,plotThreshold)   
    
""" === ENC === """
if (only == '' or only == 'enc'):
    out_path_current = out_path + 'ENC' + sep
    if (not(os.path.exists(out_path_current))):
        os.makedirs(out_path_current)
        
    enc(input_path, out_path_current, nTau, nCh,fitted_std,linear_fit_coeff, cubic_fit_coeff, plotENC)

"""=== SELF TRIGGER ===  """
if (only == '' or only == 'selfTrigger'):
    out_path_current = out_path + 'SelfTrigger' + sep
    if (not(os.path.exists(out_path_current))):
        os.makedirs(out_path_current)
        
    selfTrigger(input_path, out_path_current, nTau, nCh, plotSelfTrigger)   
    

print('Terminated in: {}'.format(time.time()-start))