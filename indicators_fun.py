import pandas as pd
import numpy as np
import datetime as dt
import os
#from util import get_data  ,plot_data
from analysis import *
import pdb
import matplotlib.pyplot as plt


def daily_returns(df):
    df1 = df.copy()
    df2 = (df1[1:]/df1[:-1].values) - 1
    return(df2)

def compound_returns(df):
    df1 = df.copy()
    df1 = df1/df1.iloc[0] - 1
    df1.iloc[0,] = 0 
    return(df1)

def normalize_data(df):
    df1 = df.copy()
    return (df1/df1.iloc[0])


def momentumfun(prices,n=14):

    salida = np.zeros_like(prices)
    salida = (prices.shift(-1)/prices)-1
    salida.iloc[:n,:]=np.nan
    return salida



def rsifunction(prices,n=14):
    #pdb.set_trace()
    deltas = prices.diff()
    up_rets = deltas[deltas>=0].fillna(0).cumsum()
    down_rets =-1*deltas[deltas<0].fillna(0).cumsum()
    
    
    up_gain = prices.copy()
    up_gain.ix[:] = 0
    up_gain.values[n:,:] = up_rets.values[n:,:] - up_rets.values[:-n,:]
    
    down_loss = prices.copy()
    down_loss.ix[:] = 0
    down_loss.values[n:,:] = down_rets.values[n:,:] - down_rets.values[:-n,:]
    
    #seed = deltas[:n+1]
    #up = seed[seed>=0].sum()/n
    #down = seed[seed<0].sum()/n
    #rs = up/down
    #rsi = np.zeros_like(prices)
    #rsi[:n] = 100. - 100./(1.+rs)

    rsi = prices.copy()
    rsi.ix[:,:]=0

    rs = (up_gain/n)/(down_loss/n)
    rsi = 100-(100/(1+rs))
    
    #for day in range(len(prices)):
    #   up            = up_gain.ix[day,:]
    #    down          = down_loss.ix[day,:]
    #    rs            = (up/n)/(down/n)
    #    rsi.ix[day,:] = 100-(100/(1+rs))

    rsi[rsi==np.inf]=100
    rsi.ix[:n,:]=np.nan
        
    
    
    
    #for i in range(n,len(prices)):
    #    delta = deltas.iloc[i-1]
    #    if delta > 0:
    #        upval = delta
    #        downval = 0
#
#        else:
#            upval = 0
#            downval = - delta
#        
#        up = (up*(n-1)+upval)/n
#        down = (down*(n-1)+downval)/n
#
#        rs = up/down
#        
#        rsi[i] = 100. - 100./(1.+rs)

    #rsi = pd.DataFrame(data=rsi , index = prices.index)
    #rsi.iloc[:n] = np.nan
    return rsi
                   
    
def plot_data(df,title='Normalized Data'):

    #fig, ax = plt.subplots()

    df1 =df['upp_std']
    
    plt.subplot(311)
    plt.plot('upp_std',data=df,linewidth=2,linestyle='--')
    plt.plot('down_std',data=df,linewidth=2,linestyle='--')
    plt.plot('moving_avarage',data=df,linewidth=2)
    plt.plot('Portfolio',data=df,linewidth=1)
    plt.subplot(312)
    plt.plot('rsi_val',data=df,linewidth=2)
    plt.subplot(313)
    plt.plot('momentum',data=df,linewidth=2)
    plt.show()

def indicators(sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,12,31), \
    syms = ['JPM'], \
    allocs=[1], \
    sv=1000000, rfr=0.0, sf=252.0,precios=0, \
    gen_plot=False):
    #pdb.set_trace()
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    prices_all = get_data(syms, dates)  # automatically adds SPY
    prices = prices_all[syms]  # only portfolio symbols
    prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    # Get daily portfolio value

   
    normalized_values = normalize_data(prices)
   
    alloced_values    = normalized_values*allocs
    pos_val           = alloced_values*sv
    port_val          = pos_val.sum(axis=1) # add code here to compute daily portfolio values
    d_returns         = daily_returns(port_val)
   
    prices_SPY = normalize_data(prices_SPY)
    port_val   = normalize_data(port_val)

#    moving_avarage    = normalized_values.rolling(14,min_periods=None,freq=None,center=False).mean()

#    rolling_std       = normalized_values.rolling(14,min_periods=None,freq=None,center=False).std()

    moving_avarage    = normalized_values.rolling(14,min_periods=None,center=False).mean()
    rolling_std       = normalized_values.rolling(14,min_periods=None,center=False).std()
    

    rsi_val           = rsifunction(normalized_values)
    upp_std           = moving_avarage + 2*rolling_std
    down_std          = moving_avarage - 2*rolling_std
    momentum          = momentumfun(normalized_values)
    moving_avarage    = normalized_values/moving_avarage
    bbp               = (normalized_values-down_std)/(upp_std-down_std)
    sma_cross         = pd.DataFrame(0,columns=moving_avarage.columns,index=moving_avarage.index)
    sma_cross[moving_avarage>=1] =1
    sma_cross[1:] = sma_cross.diff()
    sma_cross.ix[0] = 0
    rsi_spy = rsifunction(normalize_data(prices_all))
    rsi_spy = pd.DataFrame(data=rsi_spy['SPY'],index=rsi_spy.index)
   
   
   
   
    # Get portfolio statistics (note: std*_daily_ret = volatility)
    #cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        df_temp = pd.concat([port_val, prices_SPY,moving_avarage,upp_std,down_std,rsi_val,momentum], keys=['Portfolio','SPY','moving_avarage','upp_std','down_std','rsi_val','momentum'], axis=1)
        plot_data(df_temp)
       
    return (normalized_values ,bbp,moving_avarage,rsi_val,rsi_spy,momentum,sma_cross)
#indicators(gen_plot=True)
#indicators(gen_plot=True)

print('hola mundo')
