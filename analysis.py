
import pandas as pd
import numpy as np
import datetime as dt
#from util import get_data, plot_data
import matplotlib.pyplot as plt
import pdb

# This is the function that will be tested by the autograder
# The student must update this code to properly implement the functionality


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
    
def plot_data(df,title='Normalized Data'):
    ax=df.plot(title=title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.grid(linestyle='--', linewidth=1)
    plt.show()
    


def assess_portfolio(portfolio=None,sd = dt.datetime(2008,1,1), ed = dt.datetime(2009,1,1), \
    syms = ['GOOG','AAPL','GLD','XOM'], \
    allocs=[0.1,0.2,0.3,0.4], \
    sv=1000000, rfr=0.0, sf=252.0,precios=0, \
    gen_plot=False):
    # Read in adjusted closing prices for given symbols, date range
    dates = pd.date_range(sd, ed)
    if(type(portfolio)==type(None)):
        prices_all = get_data(syms, dates)  # automatically adds SPY
        prices = prices_all[syms]  # only portfolio symbols
        prices_SPY = prices_all['SPY']  # only SPY, for comparison later
    else:
        #pdb.set_trace()
        prices_all = get_data(syms,dates)
        prices = portfolio# pd.concat([portfolio,prices_all[syms]],axis=1)
        prices_SPY = prices_all[syms]
    # Get daily portfolio value
    
    
    
    if( (isinstance(precios,pd.core.series.Series))):
        pdb.set_trace()
        precios = precios.to_frame()
        normalized_values = normalize_data(precios)
        
    else:
        normalized_values = normalize_data(prices)

    
    alloced_values    = normalized_values*allocs
    pos_val           = alloced_values*sv
    port_val          = pos_val.sum(axis=1) # add code here to compute daily portfolio values

    d_returns         = daily_returns(port_val)
    cr                = port_val.iloc[-1]/port_val.iloc[0] -1
    adr               = d_returns.mean()
    sddr              = d_returns.std()
    sr                = (adr/sddr)*np.sqrt(252)
    prices_SPY = normalize_data(prices_SPY)
    port_val   = normalize_data(port_val)
    port_val   = pd.DataFrame(data=port_val,index=port_val.index,columns=['Portfolio'])
    # Get portfolio statistics (note: std_daily_ret = volatility)
    #cr, adr, sddr, sr = [0.25, 0.001, 0.0005, 2.1] # add code here to compute stats

    # Compare daily portfolio value with SPY using a normalized plot
    if gen_plot:
        # add code to plot here
        if(type(portfolio)==type(None)):
            df_temp = pd.concat([port_val, prices_SPY], keys=['Portfolio', 'SPY'], axis=1)
        else:
            #pdb.set_trace()
            df_temp = pd.concat([port_val, prices_SPY], axis=1)
        plot_data(df_temp)

    # Add code here to properly compute end value
    ev = sv

    return(cr, adr, sddr, sr, ev)






def test_code():
    # This code WILL NOT be tested by the auto grader
    # It is only here to help you set up and test your code

    # Define input parameters
    # Note that ALL of these values will be set to different values by
    # the autograder!
    start_date = dt.datetime(2010,6,1)
    end_date = dt.datetime(2010,12,31)
    symbols = ['GOOG', 'AAPL', 'GLD', 'XOM']
    allocations = [0.2, 0.3, 0.4, 0.1]
    start_val = 1000000  
    risk_free_rate = 0.0
    sample_freq = 252
    

    
    # Assess the portfolio
    cr, adr, sddr, sr, ev = assess_portfolio(sd = start_date, ed = end_date,
        syms = symbols, 
        allocs = allocations,
        sv = start_val, 
        gen_plot = True)

    # Print statistics
    print("Start Date:", start_date)
    print("End Date:", end_date)
    print("Symbols:", symbols)
    print("Allocations:", allocations)
    print("Sharpe Ratio:", sr)
    print("Volatility (stdev of daily returns):", sddr)
    print("Average Daily Return:", adr)
    print("Cumulative Return:", cr)

if __name__ == "__main__":
    test_code()
