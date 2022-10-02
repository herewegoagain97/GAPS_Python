# -*- coding: utf-8 -*-
"""
Created on Tue Aug 30 10:47:51 2022

@author: ardit
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import norm
from functions.utils import adjacent_values



def pedestal(input_path, out_path_current,nTau,nCh,plot):
    
    # setting paths
    sep = os.path.sep
    processed_path = out_path_current + 'Processed' + sep
    if (not(os.path.exists(processed_path))):
        os.makedirs(processed_path)
    file_mu = processed_path + 'mu.dat'
    file_sigma = processed_path + 'sigma.dat'
    
    # extracting the data
    df=list(range(nTau));
    for i in range(nTau):
       try:
        df[i]=pd.read_table(input_path+'Pedestals_tau{}.dat'.format(i),lineterminator="\n",
                   delim_whitespace=True,names=['Delay','DAC','Type','CH','Value'],
                   header=None,comment='#',dtype=np.float64).dropna();
        df[i]['Tau']=i;
       except:
           raise OSError;
    
    # table join and grouping     
    df_all=pd.concat(df)
    df_g=df_all.groupby(['Tau','CH'])
    df_g2=df_all.groupby(['CH','Tau'])
    
    # PROCESSING: mean between channels 
    v_mean=list(range(nTau))        
    p_mean=pd.DataFrame()
    for y in range(nTau):   
        for x in range(nCh):
            p_mean['Value_CH_{}'.format(x)]=df_g.get_group((y,x)).reset_index(drop=True)['Value']
        v_mean[y]=p_mean.mean(axis=1)
    #all_mean=pd.concat(v_mean,axis=1) 
    
    # PROCESSING: fitting gaussian
    fit_mean_par=np.ndarray((nTau,nCh)) 
    fit_std_par=np.ndarray((nTau,nCh))
    for y in range(nCh): 
       for x in range(nTau): 
           v=df_g2.get_group((y,x))['Value']
           fit_mean_par[x,y],fit_std_par[x,y]=norm.fit(v)
    
    # SAVING       
    pd.DataFrame(fit_mean_par).to_csv(path_or_buf=file_mu,sep='\t'
                                      ,index=False,header=False,float_format='%.6e')
    pd.DataFrame(fit_std_par).to_csv(path_or_buf=file_sigma,sep='\t'
                                     ,index=False,header=False,float_format='%.6e')
    
    print('Pedestal Processing complete.')    
    
    # PLOTTING
    if(plot):
        plt.ioff()
        
        # folder paths
        sep = os.path.sep
        out_path_norm = out_path_current + 'Normal' + sep
        if (not(os.path.exists(out_path_norm))):
            os.makedirs(out_path_norm)
        
        out_path_hist = out_path_current + 'Histogram' + sep
        if (not(os.path.exists(out_path_hist))):
            os.makedirs(out_path_hist)
        
        out_path_viol = out_path_current + 'Violin' + sep
        if (not(os.path.exists(out_path_viol))):
            os.makedirs(out_path_viol)

        # plot normal
        print('Plotting Pedestal Normal...')
        for y in range(nTau):
            f,ax=plt.subplots(constrained_layout=True)
            f.set_size_inches(18,8)
             
            for x in range(nCh):
                group_x=df_g.get_group((y,x))
                ax.plot(np.arange(0,1000),group_x['Value']
                         ,label="Ch#{}".format(x))
                
            chartBox = ax.get_position()
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
            ax.legend(loc='best',bbox_to_anchor=(1,1),frameon=True,ncol=2,fontsize='x-large')
            ax.set_xlabel('Time',fontsize=22)
            ax.set_ylabel('Channel Out[ADC Code]',fontsize=22)
            ax.margins(x=0,y=0.02)
            ax.grid()
            ax.set_title("Pedestal in Time for Tau {}".format(y)
                         ,fontsize=19,weight='bold')
            ax.set_ylim(0,600)
            ax.tick_params(axis='both',labelsize='x-large')
            
            plt.savefig(out_path_norm+"Normal Tau {}.pdf".format(y)
                         ,bbox_inches="tight",pad_inches=0.5)
            plt.close()
        
        # plot normal-mean
        print('Plotting Pedestal Normal-mean...')
        for y in range(nTau):
            f,ax=plt.subplots(constrained_layout=True)
            f.set_size_inches(18,8)
             
            ax.plot(np.arange(0,1000),v_mean[y],linewidth=0.75)
                
            chartBox = ax.get_position()
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
            ax.set_xlabel('Time',fontsize=22)
            ax.set_ylabel('Channel Out[ADC Code]',fontsize=22)
            ax.margins(x=0,y=0.02)
            ax.grid()
            ax.set_title("Pedestal in Time for Tau {}-Mean".format(y)
                         ,fontsize=19,weight='bold')
            ax.set_ylim(0,600)
            ax.tick_params(axis='both',labelsize='x-large')
            plt.savefig(out_path_norm+"Normal Tau {}-Mean.pdf".format(y)
                         ,bbox_inches="tight",pad_inches=0.5)
            plt.close()
            
        # plot histograms
        print('Plotting Pedestal Histograms...')
        for y in range(nCh): 
            out_path_single = out_path_hist + 'Channel_' + str(y) + sep
            if (not(os.path.exists(out_path_single))):
                os.makedirs(out_path_single)
            for x in range(nTau): 
                f,ax=plt.subplots(constrained_layout=True)
                f.set_size_inches(18,8)
                v=df_g2.get_group((y,x))['Value']
                xmin = int(np.nanmin(v)) - 10
                xmax = int(np.nanmax(v)) + 10
                
                
                bin_he,bins,_=ax.hist(v, histtype="bar",bins=(xmax-xmin+1), range=(xmin,xmax)
                        ,label=' #Tot=1000\n Mean={:.2f}\n Std={:.2f}'.format(fit_mean_par[x,y],fit_std_par[x,y])
                        ,edgecolor='black', linewidth=1.2)
                chartBox = ax.get_position()
                ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
                ax.legend(loc='best',bbox_to_anchor=(1,1),frameon=True,ncol=2,fontsize='x-large',framealpha=1)
                ax.set_xlabel('Channel Out[ADC Code]',fontsize=22)
                ax.set_ylabel('Occurrences',fontsize=22)
                ax.margins(x=0,y=0.02)
                ax.grid(axis='y')
                ax.set_title("Pedestal Hist CH {}-Tau {}".format(y,x)
                         ,fontsize=19,weight='bold')
                
                ax.plot(bins,norm.pdf(bins,loc=fit_mean_par[x,y],scale=fit_std_par[x,y])*np.trapz(bin_he), 'r--', linewidth=2)
                ax.tick_params(axis='both',labelsize='x-large')
                plt.savefig(out_path_single+"Hist CH {}_Tau{}.pdf".format(y,x)
                         ,bbox_inches="tight",pad_inches=0.5)
                         
                plt.close()

        
        # plot violin
        print('Plotting Pedestal Violins...')
        for y in range(nCh): 
            f,ax=plt.subplots(constrained_layout=True)
            f.set_size_inches(14,8)
            l=list(range(nTau))
            for x in range(nTau): 
                v=df_g2.get_group((y,x))['Value']
                l[x]=v.values
                
            parts=ax.violinplot(l,showmeans=False) 
            
            ax.set_ylabel('Channel Out[ADC Code]',fontsize=22)
            ax.set_xlabel('Peaking Time',fontsize=22)
            ax.margins(x=0.02,y=0.02)
            ax.set_xticks(range(1,9))
            ax.set_xticklabels(['PT.{}\n mu={:.2f}\n std={:.2f}'
                                .format(x,fit_mean_par[x,y],fit_std_par[x,y])for x in range(0,8)])
            ax.set_title("Pedestal Violinplot CH {}".format(y)
                         ,fontsize=19,weight='bold')  
            ax.tick_params(axis='both',labelsize='x-large')
            
            for pc in parts['bodies']:
                pc.set_edgecolor('black')
                pc.set_alpha(0.5)
                pc.set_linewidth(2.5)
                pc.set_facecolors('brown')
                
            quartile1, medians, quartile3 = np.percentile(l, [25, 50, 75], axis=1)
            whiskers = np.array([
                adjacent_values(sorted_array, q1, q3)
                for sorted_array, q1, q3 in zip(l, quartile1, quartile3)])
            whiskers_min, whiskers_max = whiskers[:, 0], whiskers[:, 1]
             
            inds = np.arange(1, len(medians) + 1)
            ax.scatter(inds, medians, marker='o', color='white', s=30, zorder=3)
            ax.vlines(inds, quartile1, quartile3, color='k', linestyle='-', lw=5)
            ax.vlines(inds, whiskers_min, whiskers_max, color='k', linestyle='-', lw=1)

            plt.savefig(out_path_viol+"Violin CH {}.pdf".format(y)
                        ,bbox_inches="tight",pad_inches=0.5)
            plt.close()
        
    return fit_std_par  
