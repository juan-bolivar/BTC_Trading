import pdb 
import pandas as pd
import numpy as np
import datetime as dt
import os
from analysis import *
import pdb

def author():
    return 'juan'

def compute_portvals(orders_df , start_val = 1000000, commission=9.95, impact=0.005):
    
    df_trades2 = orders_df.copy()
    df_trades2.sort_values('Date',inplace=True)
    df_trades2.loc[df_trades2['Order']=='BUY','Order']  = 1.0
    df_trades2.loc[df_trades2['Order']=='SELL','Order'] = -1.0
    df_trades2['Order'] = pd.to_numeric(df_trades2['Order'])
    
    df_trades2['Shares']=df_trades2['Shares']*df_trades2['Order']
    symbols = list(set(df_trades2.Symbol))
    df_trades = pd.DataFrame(columns=['Date']+symbols)
    inicio = 0

    for key,row in df_trades2.iterrows():
        df_trades.loc[inicio,row['Symbol']] = row['Shares']
        df_trades.loc[inicio,'Date'] = row['Date']
        inicio = inicio + 1
    
    df_trades = df_trades.fillna(0)

    start_date = df_trades.ix[0,'Date']
    end_date   = df_trades['Date'].iloc[-1]

    
    df_prices = get_data(symbols, pd.date_range(start_date, end_date))
    df_prices = df_prices[symbols]
    
    temp = pd.DataFrame(index=df_prices.index)
    df_trades = temp.join(df_trades.set_index('Date'),sort='True').rename_axis(None,axis=0).fillna(0)
    
    df_holdings = df_trades.copy()
    
    rv = pd.DataFrame(index=df_prices.index, data=df_prices.as_matrix(),columns=symbols)
    
    df_trades.index = pd.to_datetime(df_trades.index)
    df_trades = df_trades.mul(rv,axis='columns').dropna().fillna(0)
    df_diff = pd.DataFrame(index=df_trades.index)
    df_diff['impact']    = (abs(df_trades)*impact).sum(axis=1)
    df_diff['commission'] = (df_trades.iloc[:,:]!=0).sum(axis=1)*commission
    
    df_trades['Cash'] = -1*df_trades.sum(axis=1) - df_diff['commission'] - df_diff['impact']

    df_holdings = df_holdings.cumsum(axis=0)


    df_holdings['Cash'] =  start_val + df_trades['Cash'].cumsum(axis=0)

    temp           = df_holdings.iloc[:,:-1].copy()
    df_value       = temp.mul(df_prices,axis='columns')
    df_value['Value'] = df_value.sum(axis=1)

    df_portval = pd.DataFrame(index=list(range(len(df_value.index.get_values()))))
    #pdb.set_trace()
    df_portval['TotalValue']=0
    posicion = 0
    for key,value in df_value.iterrows():
        df_portval.ix[posicion,'TotalValue'] = df_value.ix[posicion,'Value'] + df_holdings.ix[posicion,'Cash']
        posicion = posicion + 1

    df_portval['Dates'] = df_value.index.get_values()
    
    df_portval = df_portval.drop_duplicates(subset='Dates',keep='last')
   
    df_portval = pd.DataFrame(data=df_portval.TotalValue.values,index=df_portval.Dates.values,columns=['TotalValue'])
    
    pdb.set_trace()
    #pd.concat([df_value['Value'],df_holdings['Cash'],df_portval,orders_df['Order']],axis=1)
    return df_portval


def test_code():
    

    #of = "./orders/orders2.csv"
    #df_trades = pd.read_csv(of,sep=',')
    pdb.set_trace()
    of  = "./salida.csv"
    df_trades  = pd.read_csv(of)

    sv = 1000000
    
    portvals = compute_portvals(orders_df=df_trades, start_val = sv,impact=0.005,commission=9.95)

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] 
    else:
        "warning, code did not return a DataFrame"

    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio,ev = assess_portfolio(sd=start_date,ed=end_date,allocs=[1],sv=sv,precios=portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY,ev = assess_portfolio(sd=start_date,ed=end_date,allocs=[1],sv=sv,syms=['SPY'])
    

    
    print("Date Range: {} to {}".format(start_date, end_date))
    print()
    print("Sharpe Ratio of Fund: {}".format(sharpe_ratio))
    print("Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY))
    print()
    print("Cumulative Return of Fund: {}".format(cum_ret))
    print("Cumulative Return of SPY : {}".format(cum_ret_SPY))
    print() 
    print("Standard Deviation of Fund: {}".format(std_daily_ret))
    print("Standard Deviation of SPY : {}".format(std_daily_ret_SPY))
    print()
    print("Average Daily Return of Fund: {}".format(avg_daily_ret))
    print("Average Daily Return of SPY : {}".format(avg_daily_ret_SPY))
    print()
    print("Final Portfolio Value: {}".format(portvals[-1]))

if __name__ == "__main__":
    test_code()
import pdb 
import pandas as pd
import numpy as np
import datetime as dt
import os
#from util import get_data, plot_data
from analysis import *
import pdb

def author():
    return 'juan'


def get_data(archive='COINBASE_FILTERED.csv',start_date=dt.datetime(2009,1,1) , end_date=dt.datetime(2010,1,1)):
    
    datos = pd.read_csv(archive,sep=',')
    datos.index = pd.to_datetime(datos['Timestamp'],infer_datetime_format =True,unit='s')
    salida= datos.loc[start_date:end_date]
    return salida['Weighted_Price']
    
    
    
    

def compute_portvals(orders_df , start_val = 1000000, commission=9.95, impact=0.005):
    
    df_trades2 = orders_df.copy()
    df_trades2.sort_values('Date',inplace=True)
    df_trades2.loc[df_trades2['Order']=='BUY','Order']  = 1
    df_trades2.loc[df_trades2['Order']=='SELL','Order'] = -1
    df_trades2['Shares']=df_trades2['Shares']*df_trades2['Order']
    symbols = list(set(df_trades2.Symbol))
    df_trades = pd.DataFrame(columns=['Date']+symbols)
    inicio = 0

    for key,row in df_trades2.iterrows():
        df_trades.loc[inicio,row['Symbol']] = row['Shares']
        df_trades.loc[inicio,'Date'] = row['Date']
        inicio = inicio + 1
    
    df_trades = df_trades.fillna(0)

    start_date = df_trades.ix[0,'Date']
    end_date   = df_trades['Date'].iloc[-1]

    pdb.set_trace()
    
    df_prices = get_data('COINBASE_FILTERED.csv',start_date=start_date,end_date=end_date)
    #df_prices = df_prices[symbols]
    
    temp = pd.DataFrame(index=df_prices.index)
    
    df_trades = temp.join(df_trades.set_index('Date'),sort='True').rename_axis(None,axis=0).fillna(0)
    
    df_holdings = df_trades.copy()
    
    rv = pd.DataFrame(index=df_prices.index, data=df_prices.as_matrix(),columns=symbols)
    
    df_trades.index = pd.to_datetime(df_trades.index)
    df_trades = df_trades.mul(rv,axis='columns').dropna().fillna(0)
    df_diff = pd.DataFrame(index=df_trades.index)
    df_diff['impact']    = (abs(df_trades)*impact).sum(axis=1)
    df_diff['commission'] = (df_trades.iloc[:,:]!=0).sum(axis=1)*commission
    
    df_trades['Cash'] = -1*df_trades.sum(axis=1) - df_diff['commission'] - df_diff['impact']

    df_holdings = df_holdings.cumsum(axis=0)


    df_holdings['Cash'] =  start_val + df_trades['Cash'].cumsum(axis=0)

    temp           = df_holdings.iloc[:,:-1].copy()
    df_value       = temp.mul(df_prices,axis='columns')
    df_value['Value'] = df_value.sum(axis=1)

    df_portval = pd.DataFrame(index=range(len(df_value.index.get_values())))
    #pdb.set_trace()
    df_portval['TotalValue']=0
    posicion = 0
    for key,value in df_value.iterrows():
        df_portval.ix[posicion,'TotalValue'] = df_value.ix[posicion,'Value'] + df_holdings.ix[posicion,'Cash']
        posicion = posicion + 1

    df_portval['Dates'] = df_value.index.get_values()
    
    df_portval = df_portval.drop_duplicates(subset='Dates',keep='last')
   
    df_portval = pd.DataFrame(data=df_portval.TotalValue.values,index=df_portval.Dates.values,columns=['TotalValue'])
    
    
    #pd.concat([df_value['Value'],df_holdings['Cash'],df_portval,orders_df['Order']],axis=1)
    return df_portval


def test_code():
    

    #of = "./orders/orders2.csv"
    #df_trades = pd.read_csv(of,sep=',')
    pdb.set_trace()
    of  = "./salida.csv"
    df_trades  = pd.read_csv(of)

    sv = 1000000
    
    portvals = compute_portvals(orders_df=df_trades, start_val = sv,impact=0.005,commission=9.95)

    if isinstance(portvals, pd.DataFrame):
        portvals = portvals[portvals.columns[0]] 
    else:
        "warning, code did not return a DataFrame"

    start_date = portvals.index[0]
    end_date = portvals.index[-1]
    
    cum_ret, avg_daily_ret, std_daily_ret, sharpe_ratio,ev = assess_portfolio(sd=start_date,ed=end_date,allocs=[1],sv=sv,precios=portvals)
    cum_ret_SPY, avg_daily_ret_SPY, std_daily_ret_SPY, sharpe_ratio_SPY,ev = assess_portfolio(sd=start_date,ed=end_date,allocs=[1],sv=sv,syms=['SPY'])
    

    
    #print "Date Range: {} to {}".format(start_date, end_date)
    #print
    #print "Sharpe Ratio of Fund: {}".format(sharpe_ratio)
    #print "Sharpe Ratio of SPY : {}".format(sharpe_ratio_SPY)
    #print
    #print "Cumulative Return of Fund: {}".format(cum_ret)
    #print "Cumulative Return of SPY : {}".format(cum_ret_SPY)
    #print 
    #print "Standard Deviation of Fund: {}".format(std_daily_ret)
    #print "Standard Deviation of SPY : {}".format(std_daily_ret_SPY)
    #print
    #print "Average Daily Return of Fund: {}".format(avg_daily_ret)
    #print "Average Daily Return of SPY : {}".format(avg_daily_ret_SPY)
    #print
    #print "Final Portfolio Value: {}".format(portvals[-1])
#
#if __name__ == "__main__":
    test_code()
