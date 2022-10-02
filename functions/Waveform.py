# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 10:22:03 2022

@author: ardit
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions.utils import MAD
from functions.utils import count_o


def waveform(input_path, out_path_current,nTau,nCh,plot):
    #setting path
    sep = os.path.sep
    processed_path = out_path_current + 'Processed' + sep
    if (not(os.path.exists(processed_path))):
        os.makedirs(processed_path)
    file_peak=processed_path + 'Peak_times.dat' 
 
    # extracting data
    df=list(range(nTau));
    for i in range(nTau):
       try:
        df[i]=pd.read_table(input_path+'WaveformScan_fast_tau{}.dat'.format(i),lineterminator="\n",
                   delim_whitespace=True,names=['Delay','DAC','Type','CH','Value'],
                   header=None,comment='#',dtype=np.float64).dropna();
        df[i]['Tau']=i;
       except:
           raise OSError;
           
    # PROCESSING: scaled MAD without outliers
    df_all=pd.concat(df);
    df_g=df_all.groupby(['CH','Tau','Delay']);
    
    MAD_value = df_g['Value'].transform(MAD);
    val_vs_med=df_g['Value'].transform(lambda df:np.abs(df-df.agg('median'))) ;
    v_bool=val_vs_med < (3*MAD_value);
     
    new_df = df_all[v_bool];
    df_g_out=new_df.groupby(['CH','Tau','Delay']);
    mean_out=df_g_out['Value'].agg(['mean','std']);
    mean_i=df_g['Value'].agg(['mean','std','median']); 


    df_all['outliers']=v_bool.values
    outliers_count=df_all.groupby(['CH','Tau','Delay'])['outliers'].agg(count_o);
    df_res=mean_i.join(outliers_count).join(mean_out,rsuffix='_out')
    
    # PROCESSING: peak time
    df_res['Delay_conv']=(df_res.reset_index(level=[0,1]).index.values)*20.83e-9
    oppa_2=df_res.groupby(['Tau','CH'])
    peak_t_32ch=pd.Series((oppa_2.apply(lambda x: np.argmax(x['mean'])))*20.83e-3,
                          name='peak_time')
    # SAVING 
    peak_t_32ch.to_csv(path_or_buf=file_peak,sep='\t'
                                      ,index=True,header=False,float_format='%.6e')

    print('Waveform Processing complete.')    
    
    # PLOTTING
    if(plot):   
        plt.ioff()
        
        # folder path
        sep = os.path.sep
        out_path_chann = out_path_current + 'Peak time CH' + sep
        if (not(os.path.exists(out_path_chann))):
            os.makedirs(out_path_chann)        
        
        out_path_tau = out_path_current + 'Peak time TAU' + sep
        if (not(os.path.exists(out_path_tau))):
            os.makedirs(out_path_tau)
        
        # PLOT for each Channel
        print('Plotting Waveform for channel...')
        for y in range(nCh):
            f,ax=plt.subplots(constrained_layout=True)
            f.set_size_inches(18,8)
            
            for x in range(nTau):
                
                group_x=oppa_2.get_group((x,y))
                i=format(np.argmax(group_x['mean'])*20.83e-3, ".2f")
                plt.plot(group_x['Delay_conv'],group_x['mean']
                         ,label="Tau{} (Pt_{})".format(x,i))
                
            chartBox = ax.get_position()
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
            ax.legend(loc='best',bbox_to_anchor=(1,1),frameon=True,ncol=1,fontsize='x-large')
            ax.set_xlabel('t[us]',fontsize=22)
            ax.set_ylabel('Channel_Out[ADC_Code]',fontsize=22)
            ax.margins(x=0,y=0.02)
            ax.grid()
            ax.set_title("Waveform Scan for CH {} - 1000 Dac_inj code".format(y)
                         ,fontsize=19,weight='bold')
            ax.tick_params(axis='both',labelsize='x-large')
            plt.savefig(out_path_chann+"CH-{}.pdf".format(y),bbox_inches="tight",pad_inches=0.5)            
            plt.close()
        
        # PLOT for each peak time
        print('Plotting Waveform for Tau...') 
        for y in range(nTau):
            f,ax=plt.subplots(constrained_layout=True)
            f.set_size_inches(18,8)
            
            for x in range(nCh):
                
                group_x=oppa_2.get_group((y,x))
                i=format(np.argmax(group_x['mean'])*20.83e-3, ".2f")
                ax.plot(group_x['Delay_conv'],group_x['mean']
                        ,label="Ch#{} (Pt_{})".format(x,i))
               
            chartBox = ax.get_position()
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
            ax.legend(loc='best',bbox_to_anchor=(1,1),frameon=True,ncol=2,fontsize='x-large')
            ax.set_xlabel('t[us]',fontsize=22)
            ax.set_ylabel('Channel_Out[ADC_Code]',fontsize=22)
            ax.margins(x=0,y=0.02)
            ax.grid()
            ax.set_title("Waveform Scan for Tau {} - 1000 Dac_inj code".format(y)
                         ,fontsize=19,weight='bold')
            
            ax.tick_params(axis='both',labelsize='x-large')
            plt.savefig(out_path_tau+"PT-{}.pdf".format(y),bbox_inches="tight",pad_inches=0.5)
            plt.close()