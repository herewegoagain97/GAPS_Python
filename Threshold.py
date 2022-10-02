# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 16:15:06 2022

@author: ardit
"""


import os
import numpy as np
from scipy.optimize import curve_fit
from functions.utils import err_func
import matplotlib.pyplot as plt
import pandas as pd

#%% FUNZIONE PER LA LETTURA DATI SU PATH SPECIFICATO
def threshold(input_path, out_path_current,nTau,nCh,nFthr,plot):
    
    # setting paths
    sep = os.path.sep
    processed_path = out_path_current + 'Processed' + sep
    if (not(os.path.exists(processed_path))):
        os.makedirs(processed_path)    
    
    # import dati
    df_all=list(range(nTau))
    df=list(range(nFthr))
    for i in range(nTau):
        for i2 in range (nFthr):
            df[i2]=pd.read_table(input_path+'ThresholdScan_fthr{}_tau{}.dat'.format(i2,i),lineterminator="\n",
                                    delim_whitespace=True,names=['Threshold','DAC','Events','Triggered','CH'],
                                    header=None,comment='#',dtype=np.float64).dropna()
            df[i2]['FTHR']=i2
        df_all[i]=pd.concat(df)
        df_all[i]['Tau']=i
    df_tot=pd.concat(df_all)

    
    df_tot=df_tot.assign(Percentuale=  lambda s: (s['Triggered']/s['Events']));
    df_gb=df_tot.groupby(['Tau','CH','FTHR'],as_index=False); 
    
    # PROCESSING: fitting FTHR curves
    fitted_par=pd.DataFrame()
    for i in range(nTau):
        for i2 in range(nCh):
            for i3 in range(nFthr):
                supp=df_gb.get_group((i,i2,i3))
                par,_=curve_fit(err_func, supp['Threshold'], supp['Percentuale'], p0=[220,1])
                
                supp2=pd.DataFrame(np.array([[*par,i,i2,i3]]),columns=['a','b','Tau','CH','FTHR'])
                fitted_par=pd.concat([fitted_par,supp2],axis=0)
                
    fitted_par=fitted_par.set_index(['Tau','CH','FTHR']) 
    
    # Saving
    fitted_par.to_csv(path_or_buf=processed_path+'Fit_parameters.dat',sep='\t'
                      ,index_label=['#Tau','#CH','#FTHR'],index=True
                      ,header=['#a','#b'],float_format='%.6f')
   
    print('Processing Threshold complete.')
    # PLOTTING
    if (plot):       
        plt.ioff()

        # folder paths
        sep = os.path.sep
        out_path_raw = out_path_current + 'Plot Raw CH-PT' + sep
        if (not(os.path.exists(out_path_raw))):
            os.makedirs(out_path_raw)
        
        out_path_fit = out_path_current + 'Plot Fit CH-PT' + sep
        if (not(os.path.exists(out_path_fit))):
            os.makedirs(out_path_fit)        
        
        # Grafici Threshold raw
        print('Plotting Threshold raw...')
        for i in range(nTau): 
            for i2 in range(nCh):
                f,ax=plt.subplots(constrained_layout=True)
                f.set_size_inches(18,8)
                for i3 in range(nFthr): 
                    ax.plot(df_gb.get_group((i,i2,i3))['Threshold']
                            ,df_gb.get_group((i,i2,i3))['Percentuale'],label='Fthr_{}'.format(i3))
                            
                chartBox = ax.get_position()
                ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
                ax.legend(loc='best',bbox_to_anchor=(1,1),frameon=True,ncol=1,fontsize='x-large')
                ax.set_title('Scatti % CH{} tau={}'.format(i2,i),fontsize=19,weight='bold');
                ax.grid();
                ax.set_xlabel('Threshold',loc='right',fontsize=22);
                ax.set_ylabel('% scatti',fontsize=22);
                ax.tick_params(axis='both',labelsize='x-large')
                ax.margins(x=0,y=0.02);
                plt.savefig(out_path_raw+"Raw_CH_{}-PT_{}.pdf".format(i2,i)
                            ,bbox_inches="tight",pad_inches=0.5)
                plt.close()
        # Grafici Threshold raw
        print('Plotting Threshold fitted...')
        for i in range(nTau): 
            for i2 in range(nCh): 
                f,ax=plt.subplots(constrained_layout=True)
                f.set_size_inches(18,8)
                for i3 in range(nFthr): 
                    a=fitted_par.loc[(i,i2,i3),'a']
                    b=fitted_par.loc[(i,i2,i3),'b']
                    ax.plot(np.linspace(200,256,1000)
                            ,err_func(np.linspace(200,256,1000),a,b)
                            ,label='Fthr_{} (a = {:.2f}, b = {:.2f})'.format(i3,a,b))
                            
                chartBox = ax.get_position()
                ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
                ax.legend(loc='best',bbox_to_anchor=(1,1),frameon=True,ncol=1,fontsize='x-large')
                ax.set_title('Scatti % CH{} tau={}-Fitted'.format(i2,i),fontsize=19,weight='bold');
                ax.grid();
                ax.set_xlabel('Threshold',loc='right',fontsize=22);
                ax.set_ylabel('% scatti',fontsize=22);
                ax.tick_params(axis='both',labelsize='x-large')
                ax.margins(x=0,y=0.02);
                plt.savefig(out_path_fit+"Fit_CH_{}-PT_{}.pdf".format(i2,i)
                            ,bbox_inches="tight",pad_inches=0.5)
                plt.close()