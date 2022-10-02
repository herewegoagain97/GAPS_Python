# -*- coding: utf-8 -*-
"""
Created on Sat Sep 10 09:39:42 2022

@author: ardit
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Definizioni delle costanti
def enc (input_path, out_path_current,nTau,nCh,std_ped,linear_fit_coeff,cubic_fit_coeff,plotENC):
    sep = os.path.sep
    out_path_proc = out_path_current + 'Processed' + sep
    if (not(os.path.exists(out_path_proc))):
        os.makedirs(out_path_proc)
    
    # Processing
    k=0.841*2.35
    
    filt_cubic=cubic_fit_coeff.where(cubic_fit_coeff.gt(0.5),other=np.nan).transpose()
    filt_linear=linear_fit_coeff.where(linear_fit_coeff.gt(0.5),other=np.nan).transpose()
    
    enc_lin= pd.DataFrame(np.multiply(k, np.divide(std_ped, filt_linear)))#<--- in FWHM
    enc_cub= pd.DataFrame(np.multiply(k, np.divide(std_ped, filt_cubic)))
    
    # SAVING
    enc_lin.to_csv(path_or_buf= out_path_proc+"enc_lin.dat",sep='\t'
                                      ,index=False,header=False,float_format='%.6f', na_rep="\t")
    enc_cub.to_csv(path_or_buf= out_path_proc+"enc_cub.dat",sep='\t'
                                      ,index=False,header=False,float_format='%.6f', na_rep="\t")
    print('ENC Processing complete.') 
    # PLOT
    if(plotENC):
        plt.ioff()
        
        # creating folders
        sep = os.path.sep
        out_path_plot = out_path_current + 'Plot CH' + sep
        if (not(os.path.exists(out_path_plot))):
            os.makedirs(out_path_plot)
        
        # linear plot
        print('Plotting ENC lines linear...')
        f,ax=plt.subplots(constrained_layout=True)
        f.set_size_inches(18,8)
        for i in range(nTau):
            ax.plot(np.arange(nCh),enc_lin.iloc[i], marker='*', markersize=10
                    ,linewidth=1, label='Tau {}'.format(i))
        
        ax.plot(np.linspace(-2,nCh+1,1000),np.linspace(4,4,1000),linestyle='--',color='red',label='4 KeV')#<--linea orizzontale
        ax.grid()
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
        ax.legend(loc='best',bbox_to_anchor=(1.2,0.95),frameon=True,ncol=1,fontsize='x-large')
        ax.set_xlabel('Channels',fontsize=22)
        ax.set_ylabel('FWHM ENC [KeV]',fontsize=22)
        ax.set_title("FWHM ENC [KeV](Pedestal rms/Gain linear fit slope)"
                     ,fontsize=19,weight='bold')
        ax.margins(x=-0.01,y=0.02)
        ax.tick_params(axis='both',labelsize='x-large')
        plt.savefig(out_path_plot+'enc_linear.pdf', bbox_inches="tight",pad_inches=0.5)                  
        
        # cubic plot
        print('Plotting ENC lines cubic...')
        f,ax=plt.subplots(constrained_layout=True)
        f.set_size_inches(18,8)
        for i2 in range(nTau):
            ax.plot(np.arange(nCh),enc_cub.iloc[i2], marker='*', markersize=10
                    , linewidth=1, label='Tau {}'.format(i2))
        
        ax.plot(np.linspace(-2,nCh+1,1000),np.linspace(4,4,1000),linestyle='--',color='red',label='4 KeV')#<--linea orizzontale
        ax.grid()
        chartBox = ax.get_position()
        ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
        ax.legend(loc='best',bbox_to_anchor=(1.2,0.95),frameon=True,ncol=1,fontsize='x-large')
        ax.set_xlabel('Channels',fontsize=22)
        ax.set_ylabel('FWHM ENC [KeV]',fontsize=22)
        ax.set_title("FWHM ENC [KeV](Pedestal rms/Gain cubic fit slope)"
                     ,fontsize=19,weight='bold')
        ax.margins(x=-0.01,y=0.02)
        ax.tick_params(axis='both',labelsize='x-large')
        plt.savefig(out_path_plot+'enc_cubic.pdf', bbox_inches="tight",pad_inches=0.5)        
        
        # heatmap plot linear     
        print('Plotting ENC heatmap linear...')
        f,ax_heat=plt.subplots(constrained_layout=True)
        f.set_size_inches(18,10) 
        
        pos = ax_heat.imshow(enc_lin.transpose())
        ax_heat.set_aspect('auto')
        for i in range(nCh):
            for j in range(nTau):
                ax_heat.text(j,i, "%.2f" %  enc_lin.transpose().iloc[i, j], ha='center', va='center')
        ax_heat.figure.colorbar(pos)
        ax_heat.tick_params(axis='both',labelsize='x-large')
        plt.ylabel("Channels",fontsize=22)
        plt.xlabel("$\\tau [\\mu s]$", fontsize=22)
        plt.title('FWHM ENC [keV] (Pedestal rms / Gain linear fit slope)',fontsize=19,weight='bold')          
        plt.savefig(out_path_plot+'enc_heatmap_linear.pdf', bbox_inches="tight",pad_inches=0.5) 
         
        # heatmap plot cubic
        print('Plotting ENC heatmap cubic...')
        f,ax_heat=plt.subplots(constrained_layout=True)
        f.set_size_inches(18,10) 
        
        pos = ax_heat.imshow(enc_cub.transpose())
        ax_heat.set_aspect('auto')
        for i in range(nCh):
            for j in range(nTau):
                ax_heat.text(j,i, "%.2f" %  enc_cub.transpose().iloc[i, j], ha='center', va='center')
        ax_heat.figure.colorbar(pos)
        ax_heat.tick_params(axis='both',labelsize='x-large')
        plt.ylabel("Channels",fontsize=22)
        plt.xlabel("$\\tau [\\mu s]$", fontsize=22)
        plt.title('FWHM ENC [keV] (Pedestal rms / Gain cubic fit slope)',fontsize=19,weight='bold')          
        plt.savefig(out_path_plot+'enc_heatmap_cubic.pdf', bbox_inches="tight",pad_inches=0.5)           
            
            
            
            
            