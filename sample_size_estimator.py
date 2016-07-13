import sys
import os
import operator
import copy
import re

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from collections import defaultdict


class Power_Estimator(object):

    def __init__(self, N, ctrl_frac, trt_frac, ctrl_eff, trt_eff, CL, n_bins):
        
        self._CL = CL
        self._ctrl_n = N * ctrl_frac
        self._trt_n = N * trt_frac
        
        self._ctrl_eff = ctrl_eff
        self._trt_eff = trt_eff
        
        self._pool_eff = (self._ctrl_eff * self._ctrl_n + self._trt_eff * self._trt_n) / (self._ctrl_n + self._trt_n)
 
        self._n_bins = n_bins
       
    def get_power(self, one_side = True):
        
        h0_mean = 0
        h0_sigma = np.sqrt(self._pool_eff * (1 - self._pool_eff) * (1/self._ctrl_n + 1/self._trt_n))     
        z_alpha = stats.norm.ppf(1-(1-self._CL)/self._n_bins)
        
        
        critical = h0_mean + z_alpha * h0_sigma
        
        h1_mean = self._trt_eff - self._ctrl_eff
        h1_sigma = np.sqrt(self._ctrl_eff * (1 - self._ctrl_eff) / self._ctrl_n + 
                           self._trt_eff * (1 - self._trt_eff) / self._trt_n)
        
        power = 1 - stats.norm.cdf(-1 * (h1_mean - critical) / h1_sigma)
        
        result = {}
        
        result['power'] = power
        result['h0_mean'] = h0_mean
        result['h0_sigma'] = h0_sigma
        result['z_alpha'] = z_alpha
        result['h1_mean'] = h1_mean
        result['h1_sigma'] = h1_sigma
        result['z_beta'] = (h1_mean - critical) / h1_sigma      
        
        return(result)


def get_threshold(trt_frac, test_power):

    first_90 = True
    first_80 = True
    first_70 = True

    threshold_90 = 0
    threshold_80 = 0
    threshold_70 = 0

    for x, power in zip(2 * trt_frac * 100, np.array(test_power) * 100):
        if power >= 90 and first_90 == True:
            threshold_90 = x
            first_90 = False
        if power >= 80 and first_80 == True:
            threshold_80 = x
            first_80 = False
        if power >= 70 and first_70 == True:
            threshold_70 = x
            first_70 = False
    return(threshold_70, threshold_80, threshold_90)





def main():

    total = 120000
    n_bins = 6
    excl_frac = 0.1

    ctrl = 0.32

    variation = {}
    variation['First'] = 1.1    
    variation['Second'] = 1.2

    trt_frac = np.linspace(0, 1, num=1000)
    # Half of the test x will be allocated to down shift, and half up => control is 1 - 2*x
    #ctrl_frac = 1 - 2*x 
    ctrl_frac = 0.45

    power = defaultdict(list)

    global_CL = 0.95
    
    for x in trt_frac:

        power_1st = Power_Estimator(total * (1 - excl_frac), ctrl_frac / n_bins, x / n_bins, ctrl, ctrl * variation['First'], global_CL, n_bins) 
        power['First'].append(power_1st.get_power(one_side = True)['power'])
        
        power_2nd = Power_Estimator(total * (1 - excl_frac), ctrl_frac / n_bins, x / n_bins, ctrl, ctrl * variation['Second'], global_CL, n_bins) 
        power['Second'].append(power_2nd.get_power(one_side = True)['power'])


    threshold = {}

    threshold['First_70p'] = get_threshold(trt_frac, power['First'])[0]
    threshold['First_80p'] = get_threshold(trt_frac, power['First'])[1]
    threshold['First_90p'] = get_threshold(trt_frac, power['First'])[2]
    
    threshold['Second_70p'] = get_threshold(trt_frac, power['Second'])[0]
    threshold['Second_80p'] = get_threshold(trt_frac, power['Second'])[1]
    threshold['Second_90p'] = get_threshold(trt_frac, power['Second'])[2]
    
    print("For a +/- %.0f%% variation at 80%% power, need %.0f%% of population in the treatment group." % 
          ((variation['First']-1)*100, threshold['First_80p']))
    print("For a +/- %.0f%% variation at 80%% power, need %.3f%% of population in the treatment group." % 
          ((variation['Second']-1)*100, threshold['Second_80p'])) 
       
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5.5)
    ax.plot(2 * trt_frac * 100, np.array(power['First']) * 100, linewidth = 2, color = 'red', label = 'Test 1: +/- 10% pricing variation')
    ax.plot(2 * trt_frac * 100, np.array(power['Second']) * 100, linewidth = 2, color = 'blue', label = 'Test 2: +/- 20% pricing variation')
    ax.set_xlabel('X% allocated to the pricing test group', fontsize = 12)
    ax.set_ylabel('Power of test', fontsize = 12)
    ax.set_title('Power of test at 95% CL (control group size fixed at 45%)')
    #ax.axhline(y=90, linestyle='--', color='black', linewidth=1)
    ax.axhline(y=80, linestyle='--', color='black', linewidth=1)
    #ax.axhline(y=70, linestyle='--', color='black', linewidth=1)
    
    #ax.axvline(x=threshold['First_90p'], linestyle='--', color='black', linewidth=1)
    ax.axvline(x=threshold['First_80p'], linestyle='--', color='black', linewidth=1)
    #ax.axvline(x=threshold['First_70p'], linestyle='--', color='black', linewidth=1)
    
    ax.axvline(x=threshold['Second_80p'], linestyle='--', color='black', linewidth=1)
    
    ax.set_xlim(0, 55)
    
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc = 'best', fontsize = 12)
    plt.savefig('size_estimation.png')


    

    pdf_power = Power_Estimator(total * (1 - excl_frac), 0.45 / n_bins, (0.07007 / 2) / n_bins, ctrl, ctrl * variation['Second'], global_CL, n_bins)
    pdf_power.get_power(one_side = True)['power']
    
    h0 = np.random.normal(pdf_power.get_power(one_side = True)['h0_mean'], pdf_power.get_power(one_side = True)['h0_sigma'], 100000)
    h1 = np.random.normal(pdf_power.get_power(one_side = True)['h1_mean'], pdf_power.get_power(one_side = True)['h1_sigma'], 100000)
    
    fig, ax = plt.subplots()
    fig.set_size_inches(8, 5.5)

    n1, bins1, patches1 = ax.hist(x = h0, normed = True, bins = 100, color = 'blue', label = r'H0: $\mu_{0}$ = $\epsilon_{test} - \epsilon_{control}$ = 0', histtype='step')
    n2, bins2, patches2 = ax.hist(x = h1, normed = True, bins = 100, color = 'red', label = r'H1: $\mu_{1}$ = $\epsilon_{test} - \epsilon_{control}$ = 0.064', histtype='step')

    print(pdf_power.get_power(one_side = True)['z_alpha'])
    print(pdf_power.get_power(one_side = True)['h0_sigma'])

    ax.axvline(x=pdf_power.get_power(one_side = True)['h0_mean'] + pdf_power.get_power(one_side = True)['z_alpha'] * pdf_power.get_power(one_side = True)['h0_sigma'], linestyle='--', color='black', linewidth=1)

    critical1 = bins1 >= pdf_power.get_power(one_side = True)['h0_mean'] + pdf_power.get_power(one_side = True)['z_alpha'] * pdf_power.get_power(one_side = True)['h0_sigma'] - (min(bins2)+max(bins2))/(len(bins2) - 1) 
    critical2 = bins2 >= pdf_power.get_power(one_side = True)['h0_mean'] + pdf_power.get_power(one_side = True)['z_alpha'] * pdf_power.get_power(one_side = True)['h0_sigma'] - (min(bins2)+max(bins2))/(len(bins2) - 1) 

    b1 = [(bins1[critical1][i] + bins1[critical1][i+1])/2 for i, b in enumerate(bins1[critical1[:-1]])]
    b2 = [(bins2[critical2][i] + bins2[critical2][i+1])/2 for i, b in enumerate(bins2[critical2[:-1]])]

    ax.fill_between(b1, np.repeat(0, len(bins1[critical1[:-1]])), n1[critical1[:-1]], facecolor='blue', alpha=0.6)
    ax.fill_between(b2, np.repeat(0, len(bins2[critical2[:-1]])), n2[critical2[:-1]], facecolor='red', alpha=0.4)
    ax.set_xlabel(r'$\mu$', fontsize = 16)
    ax.set_ylabel('Density', fontsize = 12)
    ax.set_ylim(0, 30)

    ax.set_title('Test statistic distribution for "Test 2" (+/- 20% price variation)', fontsize = 14)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc = 'upper left', fontsize = 12)
    plt.savefig('significance_level_and_power.png')


if __name__ == '__main__':
    main()
