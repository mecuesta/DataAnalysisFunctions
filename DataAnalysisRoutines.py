# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 20:08:29 2016

@author: Manuel
"""

import pandas as pd
import numpy as np
from Voyager import *

## script to calculate PVI

def drop_outliers(df, left_limit, right_limit, plus_minus, std_factor):
# make a copy to not overwrite the original
    df = df.copy()
# conditionally check all values in df to be out of specified range
    df[(df < left_limit) | (df > right_limit)] = np.NaN
# run through df in specified ranges set by plus_minus and then check all values 
# in range for data over specified factors of standard deviation
    for i in range(plus_minus, df.size-plus_minus-1, 2*plus_minus):
        IDX = np.s_[i-plus_minus:i+plus_minus]
        df[IDX][(np.abs(df[IDX]-df[IDX].mean())) >= (std_factor*df[IDX].std())] = np.NaN
    return df



# PVI method
def PVI(ax,ay,az,lag):
# take the three factors of mag_field and find the derivatives
    dax=ax-ax.shift(-lag)
    day=ay-ay.shift(-lag)
    daz=az-az.shift(-lag)
    #dax.replace(to_replace=np.NaN, value = 0, inplace=True)
    #day.replace(to_replace=np.NaN, value = 0, inplace=True)
    #daz.replace(to_replace=np.NaN, value = 0, inplace=True)
# calc the magnitude of the derivatives of mag_field and return the pvi
    mag = dax.pow(2)+day.pow(2)+daz.pow(2)
    return mag.div(mag.mean()).pow(.5)
    


def structure_function(ax, ay, az, ar_lags, ar_powers):
# run through ar_lags and ar_powers so that for each power, run through each lag
	d = {}
	for i in ar_powers:
		array = []
		for l in ar_lags:
			dax=ax-ax.shift(-l)
			day=ay-ay.shift(-l)
			daz=az-az.shift(-l)
			strct = (dax.pow(2)+day.pow(2)+daz.pow(2)).pow(0.5).pow(i).mean()
			array += [strct]
		d[str(i)] = array
#		d[str(i)+' lags'] = ar_lags

	df = pd.Series(d)
	return df














#   dictionary = {}
#   ctr = 0
#   for i in ar_powers:
#       struct = [[]];
#       for l in ar_lags:
#       # calc the increments of all three arrays
#           dax=ax-ax.shift(-l)
#           day=ay-ay.shift(-l)
#           daz=az-az.shift(-l)
#   for k in [dax,day,daz]:
#       	    k.replace(to_replace=np.NaN,value=0, inplace=True)
#   #Calc the magnitude of the field
#   mag = (dax.pow(2)+day.pow(2)+daz.pow(2)).pow(0.5)
#   # calc the mean of the specified orders in ar_powers
#   mean_mag_power = mag.pow(i).mean()
#   # add structure of certain lag to struct
# 		struct[ctr] += [mean_mag_power]   
#   # add struct to dictionary and reset struct
#   dictionary[str(i) + 'order'] = struct[ctr]
#	 ctr += 1;
#   df = pd.DataFrame(dictionary)
#   return df
    
    
def kurtosis_function(series, lag_arr):
    
    
    kurtosis = []
    for i in lag_arr:
	# Analysis of Kurtosis
        temp = series.shift(-i)
        temp = temp.pow(4).mean()/(temp.pow(2).mean()**2)
        kurtosis += [temp]
    kurtosis = pd.Series(kurtosis)
    return kurtosis
    



def pdf_function(series, binsize):
# find rms, create empty array of arrays for bins
    series = series.copy()
    rmsval = series.std()
    series = series.div(rmsval)
# find the rms value of the series
#dropna and then sort the data from min to max, then reset the indices
    #series.dropna(inplace=True)
    series.dropna(inplace=True)
    series.sort_values(inplace=True)
    series.reset_index(drop=True, inplace = True)
    length = len(series)/binsize
    pdf = np.zeros(length); bins=np.zeros(length)
# For each bin, take the size, divide by the max-min of that bin, then add to pdf
    acc = 0
    for i in range(binsize,len(series),binsize):
      	temp = series[i-binsize:i]
        bins[acc] = temp.mean()
        pdf[acc]  = binsize/(temp.max()-temp.min())
      	acc += 1
# return array of pdfs
    return bins,pdf/len(series)


def waiting_time_routine(series, time_var, ar_pvi):
    series = series.copy()
    times = []
    indices = []
    for pvi in ar_pvi:
#tracker for the indices of the multiple occurences of the pvi
        idx = 0
        for i in series:
            if i == pvi:
# return the timestamp, not sure why, i never use it
                times += [series[timevar][idx]]
# record the index
                indices += [idx]
            idx += 1
    
    dictionary = {'idx':np.asarray(indices), 'times':np.asarray(times)}

    return pd.DataFrame(dictionary)


def get_dist_from_times_with_avg_speed(series, ar_time, ar_idx):
    avg_vels = []
    sec_times = []
    ar_dist = []
    for i in range(ar_idx) -1:
# temp for the data points bewteen the two occurences of the same pvi
        temp = series[ar_idx[i]:ar_idx[i+1]]
# find the avg velocity between the two occurences of the same pvi
        avg_vels += [temp.mean()]
# find the time in seconds between the two occurences of the same pvi
        sec_times += [(ar_idx[i+1]-ar_idx[i])*2]
# calculate the distance
        ar_dist += [avg_vels[i]*sec_times[i]]

    dictionary = {'vels':np.asarray(avg_vels), 'secs':np.asarray(sec_times), 'dist':np.asarray(ar_dist)}
    
    return pd.DataFrame(dictionary)
    
    
        
    
    
    
def correlation_coefficient(lagged_x, y, lag):
# Take copies of the time series
    lagged_x = lagged_x.copy()
    y = y.copy()
# take the shift of the spec'd series
    x = lagged_x.shift(lag)
    x.replace(to_replace=np.NaN, value=0, inplace = True)
# take the covariance bewteen the two series
    covar = x.cov(y)
# returned normalized covar by the product of the stds of both series
    return covar/(lagged_x.std()*y.std())
    

def correlation_function(lagged_x, y, ar_lags):
    corr = np.zeros(np.size(ar_lags))
    acc = 0
    for i in range(np.size(ar_lags)):
	corr[acc] = correlation_coefficient(lagged_x, y, i)
	acc += 1
    return corr


def auto_correlation(series, ar_lags):
    return correlation_function(series, series, ar_lags)



def convert_cdfs_to_dataframe(filelist, varlist, time_var_str):
    from spacepy import pycdf
    from delorean import Delorean
    
#create empty numpy arrays
    ll=len(varlist); varsdata=[np.zeros(1) for i in range(ll+1)]
# read data from cdf files and append the arrays.
    for i in filelist:
      #  print 'reading file '+i
        d = pycdf.CDF(i)
        for j in varlist:
	          idx=varlist.index(j)
	          varsdata[idx]= np.append(varsdata[idx], pycdf.VarCopy(d[j]))
   #print 'Done reading data'
#   For create an epoch array from time_var_str
#   (s)econds (s)ince (epoch) == ssepoch
    idxe = varlist.index(time_var_str); ldata=len(varsdata[0]); ssepoch=np.zeros(ldata)
    for i in range(1,ldata):
    	ssepoch[i] = Delorean(varsdata[idxe][i],timezone="UTC").epoch 
# drop the first zero before creating the data frame
    dictionary = {}; dictionary['time']=ssepoch[1:]
    for j in varlist:
        if j == time_var_str:
            dictionary['datetime']=varsdata[varlist.index(j)][1:]
        else:
            dictionary[j] = varsdata[varlist.index(j)][1:]
# create a pandas dataframe
#    dictionary = {'dist':varsdata[0]}
    d = pd.DataFrame(dictionary)
# drop the missing data values (999. for Voyager) by np.NaN
    d.replace(to_replace=999.,value=np.NaN,inplace=True)
    return d


def plot_structure_function(structured_data, ar_lags, power):
    plot(ar_lags, structured_data[str(power) + 'order'])




def curve_fitting_poly(x, y, power):
    z = np.polyfit(x, y, power)
    f = np.poly1d(z)

    y_new = f(x)

    return x, y, y_new, z






        


