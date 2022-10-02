# -*- coding: utf-8 -*-
"""
Created on Wed Sep  7 09:57:55 2022

@author: ardit
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from functions.utils import fit_linear
from functions.utils import fit_cubic

from scipy.optimize import curve_fit


def transfer(input_path, out_path_current, nCh, nTau, plot):
    
    # setting paths
    sep = os.path.sep
    processed_path = out_path_current + 'Processed' + sep
    if (not(os.path.exists(processed_path))):
        os.makedirs(processed_path)
    
    df=list(range(nTau));
    for i in range(nTau):
       try:
        df[i]=pd.read_table(input_path+'TransferFunction_fast_tau{}.dat'.format(i),lineterminator="\n",
                   delim_whitespace=True,names=['Delay','DAC','Type','CH','Value'],
                   header=None,comment='#',dtype=np.float64).dropna();
        df[i]['Tau']=i;
       except:
           raise OSError;
    
    # PROCESSING: mean value
    df_mean=list(range(nTau))
    for z in range(nTau):
        df_mean[z]=df[z].groupby(['CH','DAC']).mean().reset_index(level='DAC').groupby('CH')

    # PROCESSING: fitted coefficients 
    coeff_high_lin1=pd.DataFrame()
    coeff_high_lin2=pd.DataFrame()
    coeff_low_lin1=pd.DataFrame()
    coeff_low_lin2=pd.DataFrame()
    coeff_low_cub1=pd.DataFrame()
    coeff_low_cub2=pd.DataFrame()
    coeff_low_cub3=pd.DataFrame()
    coeff_low_cub4=pd.DataFrame()
    for i in range (nCh):#ch
        coeff_1lin1=pd.DataFrame(index=['CH_{}'.format(i)])
        coeff_2lin1=pd.DataFrame(index=['CH_{}'.format(i)])
        coeff_1lin2=pd.DataFrame(index=['CH_{}'.format(i)])
        coeff_2lin2=pd.DataFrame(index=['CH_{}'.format(i)])
        coeff_cub1=pd.DataFrame(index=['CH_{}'.format(i)])
        coeff_cub2=pd.DataFrame(index=['CH_{}'.format(i)])
        coeff_cub3=pd.DataFrame(index=['CH_{}'.format(i)])
        coeff_cub4=pd.DataFrame(index=['CH_{}'.format(i)])
        for i2 in range (nTau):#tau
            g=df_mean[i2].get_group(i)
            x=g.loc[(g['DAC']>=20000) & (g['DAC']<=60000),'DAC']
            y=g.loc[(g['DAC']>=20000) & (g['DAC']<=60000),'Value']
            popt,_=curve_fit(fit_linear, x,y)
            coeff_1lin1.insert(i2,'Tau_{}'.format(i2),popt[0])
            coeff_2lin1.insert(i2,'Tau_{}'.format(i2),popt[1])
            x=g.loc[(g['DAC']>=10)&(g['DAC']<=500) ,'DAC']
            y=g.loc[(g['DAC']>=10)&(g['DAC']<=500) ,'Value'] 
            popt,_=curve_fit(fit_linear, x,y)
            popt2,_=curve_fit(fit_cubic,x,y)
            coeff_1lin2.insert(i2,'Tau_{}'.format(i2),popt[0])
            coeff_2lin2.insert(i2,'Tau_{}'.format(i2),popt[1])
            coeff_cub1.insert(i2,'Tau_{}'.format(i2),popt2[0])
            coeff_cub2.insert(i2,'Tau_{}'.format(i2),popt2[1])
            coeff_cub3.insert(i2,'Tau_{}'.format(i2),popt2[2])
            coeff_cub4.insert(i2,'Tau_{}'.format(i2),popt2[3])
       
        coeff_high_lin1=pd.concat([coeff_high_lin1,coeff_1lin1])
        coeff_high_lin2=pd.concat([coeff_high_lin2,coeff_2lin1])
        coeff_low_lin1=pd.concat([coeff_low_lin1,coeff_1lin2])
        coeff_low_lin2=pd.concat([coeff_low_lin2,coeff_2lin2])
        coeff_low_cub1=pd.concat([coeff_low_cub1,coeff_cub1])
        coeff_low_cub2=pd.concat([coeff_low_cub2,coeff_cub2])
        coeff_low_cub3=pd.concat([coeff_low_cub3,coeff_cub3])
        coeff_low_cub4=pd.concat([coeff_low_cub4,coeff_cub4])
        
    # SAVING       
    coeff_high_lin1.to_csv(path_or_buf=processed_path+'coeff_high_lin1.dat',sep='\t'
                           ,index=False,header=False,float_format='%.6e')
    coeff_high_lin2.to_csv(path_or_buf=processed_path+'coeff_high_lin2.dat',sep='\t'
                           ,index=False,header=False,float_format='%.6e')
    coeff_low_lin1.to_csv(path_or_buf=processed_path+'coeff_low_lin1.dat',sep='\t'
                           ,index=False,header=False,float_format='%.6e')
    coeff_low_lin2.to_csv(path_or_buf=processed_path+'coeff_low_lin2.dat',sep='\t'
                           ,index=False,header=False,float_format='%.6e')
    coeff_low_cub1.to_csv(path_or_buf=processed_path+'coeff_high_cub1.dat',sep='\t'
                           ,index=False,header=False,float_format='%.6e')
    coeff_low_cub2.to_csv(path_or_buf=processed_path+'coeff_high_cub2.dat',sep='\t'
                           ,index=False,header=False,float_format='%.6e')
    coeff_low_cub3.to_csv(path_or_buf=processed_path+'coeff_high_cub3.dat',sep='\t'
                           ,index=False,header=False,float_format='%.6e')
    coeff_low_cub4.to_csv(path_or_buf=processed_path+'coeff_high_cub4.dat',sep='\t'
                           ,index=False,header=False,float_format='%.6e')    
    print('Transfer Processing complete.')    
    
    # PLOTTING
    if(plot):
        plt.ioff()
        
        # folder paths
        sep = os.path.sep
        out_path_trans = out_path_current + 'Transfer CHANNEL' + sep
        if (not(os.path.exists(out_path_trans))):
            os.makedirs(out_path_trans)
         
        out_path_trans2 = out_path_current + 'Transfer TAU' + sep
        if (not(os.path.exists(out_path_trans2))):
            os.makedirs(out_path_trans2)  
            
        out_path_trans3 = out_path_current + 'High Energy' + sep
        if (not(os.path.exists(out_path_trans3))):
            os.makedirs(out_path_trans3)
            
        out_path_trans4 = out_path_current + 'Low Energy' + sep
        if (not(os.path.exists(out_path_trans4))):
            os.makedirs(out_path_trans4)
        
        # plot for CHANNEL
        print('Plotting Transfer function fixed CH...')
        
        for x in range (nCh):
            f,ax=plt.subplots(constrained_layout=True)
            f.set_size_inches(18,8)   
            for y in range(nTau):
               ax.plot(df_mean[y].get_group(x)['DAC'],df_mean[y].get_group(x)['Value'],label="Tau {}".format(y))
               
            chartBox = ax.get_position()
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
            ax.legend(loc='best',bbox_to_anchor=(1,1),frameon=True,fontsize='x-large')
            ax.set_xlabel('Cal Voltage [DAC Inj Code]',fontsize=22)
            ax.set_ylabel('Channel Out[ADC Code]',fontsize=22)
            ax.margins(x=0,y=0.02)
            ax.grid()
            ax.set_title("Transfer Function of CH {}".format(x)
                         ,fontsize=19,weight='bold')
            ax.set_ylim(ymin=0)
            ax.tick_params(axis='both',labelsize='x-large')
            plt.savefig(out_path_trans+"Transfer CH {}.pdf".format(x)
                        ,bbox_inches="tight",pad_inches=0.5)
            plt.close()
                
                
        
        # plot for TAU
        print('Plotting Transfer function fixed TAU...')
        
        for x in range (nTau):
            f,ax=plt.subplots(constrained_layout=True)
            f.set_size_inches(18,8)   
            for y in range(nCh):
               ax.plot(df_mean[x].get_group(y)['DAC'],df_mean[x].get_group(y)['Value'],label="CH {}".format(y))
               
            chartBox = ax.get_position()
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
            ax.legend(loc='best',bbox_to_anchor=(1,1),frameon=True,ncol=2,fontsize='x-large')
            ax.set_xlabel('Cal Voltage [DAC Inj Code]',fontsize=22)
            ax.set_ylabel('Channel Out[ADC Code]',fontsize=22)
            ax.margins(x=0,y=0.02)
            ax.grid()
            ax.set_title("Transfer Function for Tau {}".format(x)
                         ,fontsize=19,weight='bold')
            ax.set_ylim(ymin=0)
            ax.tick_params(axis='both',labelsize='x-large')
            plt.savefig(out_path_trans2+"Transfer TAU {}.pdf".format(x)
                        ,bbox_inches="tight",pad_inches=0.5)
            plt.close()
        
        
        
        # plot high gain energy fit
        print('Plotting Transfer High energy...')
        
        for i in range (nCh):
            f,ax=plt.subplots(constrained_layout=True)
            f.set_size_inches(18,8)
            for i2 in range (nTau):
                g=df_mean[i2].get_group(i)
                x=g.loc[(g['DAC']>=20000) & (g['DAC']<=60000),'DAC']
                y=g.loc[(g['DAC']>=20000) & (g['DAC']<=60000),'Value']
  
                line1=ax.plot(x,y,'o',markersize=9.5
                              ,label="Tau {}\nFit: ${:.2e}X$+{:.1f} ".format(i2,coeff_high_lin1.iat[i,i2],coeff_high_lin2.iat[i,i2])
                             ,markeredgecolor='black',markeredgewidth=1.5)
                ax.plot(x,fit_linear(x,coeff_high_lin1.iat[i,i2],coeff_high_lin2.iat[i,i2]),'--',linewidth=1.7
                        ,color=line1[0].get_color())
                
            chartBox = ax.get_position()
            ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
            ax.legend(loc='best',bbox_to_anchor=(1,1),frameon=True,ncol=1,fontsize='x-large')
            ax.set_xlabel('Cal Voltage [DAC Inj Code]',fontsize=22)
            ax.set_ylabel('Channel Out[ADC Code]',fontsize=22)
            ax.margins(x=0.05,y=0.02)
            ax.grid()
            ax.set_title("High Energy Gain for CH {}".format(i)
                         ,fontsize=19,weight='bold')
            
            ax.tick_params(axis='both',labelsize='x-large')
            plt.savefig(out_path_trans3+"TF High Energy CH {}.pdf".format(i)
                        ,bbox_inches="tight",pad_inches=0.5)
            plt.close()
        
        
        
        # plot low gain energy fit 
        print('Plotting Transfer Low energy...')
        
        for i in range (nCh):
            out_path_single = out_path_trans4 + 'Channel_' + str(i) + sep
            if (not(os.path.exists(out_path_single))):
                os.makedirs(out_path_single)
            for i2 in range (nTau): 
                f,ax=plt.subplots(constrained_layout=True)
                f.set_size_inches(18,8)
                g=df_mean[i2].get_group(i)
                x=g.loc[(g['DAC']>=10)&(g['DAC']<=500) ,'DAC']
                y=g.loc[(g['DAC']>=10)&(g['DAC']<=500) ,'Value']
                
                lines=ax.plot(x,y,'o',x,fit_linear(x,coeff_low_lin1.iat[i,i2],coeff_low_lin2.iat[i,i2]),np.linspace(10,501,100)
                        ,fit_cubic(np.linspace(10,501,100),coeff_low_cub1.iat[i,i2],coeff_low_cub2.iat[i,i2],coeff_low_cub3.iat[i,i2],coeff_low_cub4.iat[i,i2]),'--'
                        ,markersize=9.5,markeredgecolor='black',markeredgewidth=1.5,linewidth=1.7,)
                chartBox = ax.get_position()
                ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.6, chartBox.height])
                ax.legend(lines,['data [10-500 DAC inj Code]','Linear interpolation\n${:+.2e}X${:+.1f}'
                                 .format(coeff_low_lin1.iat[i,i2],coeff_low_lin2.iat[i,i2])
                                 ,'Cubic interpolation:\n${:+.2e}X^3$${:+.2e}X^2$${:+.3f}X${:+.1f}'
                                 .format(coeff_low_cub1.iat[i,i2],coeff_low_cub2.iat[i,i2],coeff_low_cub3.iat[i,i2],coeff_low_cub4.iat[i,i2])]
                          ,loc='best',bbox_to_anchor=(1,1),frameon=True,ncol=1,fontsize='x-large')
                ax.set_xlabel('Cal Voltage [DAC Inj Code]',fontsize=22)
                ax.set_ylabel('Channel Out[ADC Code]',fontsize=22)
                ax.margins(x=0.05,y=0.02)
                ax.grid()
                ax.set_title("Low Energy Gain for CH {} [Tau {}]".format(i,i2)
                             ,fontsize=19,weight='bold')
                
                ax.tick_params(axis='both',labelsize='x-large')
                plt.savefig(out_path_single+"TF Low Energy CH {}-TAU {}.pdf".format(i,i2)
                            ,bbox_inches="tight",pad_inches=0.5)
                plt.close()
        
        
    
    
    return coeff_low_lin1,coeff_low_cub3 