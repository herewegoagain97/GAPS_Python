# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 17:37:51 2022

@author: ardit
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm

def selfTrigger(input_path,out_path_current,nTau,nCh,plot):
    df=list(range(nCh));
        
    for i in range(nCh):
       try:
        df[i]=pd.read_table(input_path+'SelfTrigger_ch{}.dat'.format(i),lineterminator="\n",
                       delim_whitespace=True,names=['Delay','DAC','Type','CH','Value'],
                       header=None,comment='#',dtype=np.float64,usecols=[1,2,3,4]).dropna();
        
       except:
        raise OSError;
    
    # PROCESSING    
    dummy=pd.concat(df)
    data=dummy.where(dummy['Type']==10).groupby('CH')
    
    # PLOTTING
    if(plot):
        plt.ioff()
        
        #folder
        sep = os.path.sep
        out_path_hist = out_path_current + 'Histogram' + sep
        if (not(os.path.exists(out_path_hist))):
            os.makedirs(out_path_hist)
        
        # plots
        print("Plotting Self Trigger histograms...")
        for i in range(nCh): 
            f,ax=plt.subplots(constrained_layout=True)
            f.set_size_inches(18,8)
            v=data.get_group(i)['Value']
            xmin = int(np.nanmin(v)) - 10
            xmax = int(np.nanmax(v)) + 10
            
            fit_m,fit_std=norm.fit(v)
            bin_he,bins,_=ax.hist(v, histtype="bar",bins=(xmax-xmin+1), range=(xmin,xmax)
                    ,label=' #Tot=100\n Mean={:.2f}\n Std={:.2f}'.format(fit_m,fit_std)
                    ,edgecolor='black', linewidth=1.2)
        
            chartBox = ax.get_position()
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
            ax.legend(loc='best',bbox_to_anchor=(1,1),frameon=True,ncol=2,fontsize='x-large',framealpha=1)
            ax.set_xlabel('Channel Out[ADC Code]',fontsize=22)
            ax.set_ylabel('Occurrences',fontsize=22)
            ax.margins(x=0,y=0.02)
            ax.grid(axis='y')
            ax.set_title("Self Trigger CH {}".format(i)
                     ,fontsize=19,weight='bold')
            ax.plot(bins,norm.pdf(bins,loc=fit_m,scale=fit_std)*np.trapz(bin_he), 'r--', linewidth=2)
            ax.tick_params(axis='both',labelsize='x-large')
            plt.savefig(out_path_hist+"CH {}".format(i), bbox_inches="tight",pad_inches=0.5)
            plt.close()