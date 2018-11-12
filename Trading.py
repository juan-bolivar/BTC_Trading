"""

Template for implementing StrategyLearner  (c) 2016 Tucker Balch

"""

import datetime as dt
import pandas as pd
import random
import matplotlib.pyplot as plt
import numpy as np
import pdb
import pandas as pd
import time
import tensorflow as tf
from marketsimcode import compute_portvals
from DQN import *
from tensorflow import keras
from indicators_fun import indicators
from sklearn.model_selection import train_test_split
from indicators_bitcoin import indicators

from os import listdir
import os

class StrategyLearner(object):   
    def __init__(self, verbose=False, impact=0.005 , commission=9.95):
        self.verbose = verbose
        self.impact = impact
        self.commission = commission
 

    def calculate_reward(self, holdings_actions_1 , action ,key):

        prices_all = self.prices_all
        
        holdings_actions              = {0:1,1:0.5,2:0,3:-0.5,4:-1}[action]
        
                
        if( key == prices_all.index[0]):
            
            holdings_diff             = 0
            
        else:
            price_t                   = prices_all.iloc[prices_all.index.get_loc(key)-1]
            price_t_plus_1            = prices_all.loc[key]
            cash                      = -1*(holdings_actions-holdings_actions_1) * price_t
            holdings_diff             = holdings_actions * price_t_plus_1 - holdings_actions_1 * price_t + cash
            
        if(holdings_actions_1 - holdings_actions)!=0:
            
            holdings_diff     = holdings_diff - self.commission
            
        reward = holdings_diff
        
        
        return (reward,holdings_actions)
                
    def addEvidence(self, symbol = "IBM", \
        sd=dt.datetime(2015,1,1), \
        ed=dt.datetime(2017,1,1), \
        sv = 1000000): 
        btc  = pd.read_csv('COINDESK_LAST_MONTH.CSV')
        #btc = pd.read_csv('COINBASE_FILTERED.CSV')   # CHANGE TO DATABASE
        size = int(len(btc))
        #size= int(len(btc)*0.005)
        
        btc = btc.iloc[-3*size:-size]
        
        btc[btc.columns.values] = btc[btc.columns.values].ffill()
        
        btc['TR'] =  0 
        
        a = btc['High']-btc['Low']
        b = btc['Low']-btc['Close'].shift(-1)
        c = btc['High']-btc['Close'].shift(-1)
        
        btc['TR'] = pd.concat([a,b,c],axis=1).max(axis=1)
        btc['ATR'] = btc['TR'].ewm(span = 10).mean()
        
        btc['Delta'] = btc['Close'] - btc['Open']
        
        btc['to_predict'] = btc['Delta'].apply(lambda x : 1 if(x>0) else 0)
        
        #btc.index = pd.to_datetime(btc['Timestamp'],infer_datetime_format =True,unit='s')
        btc.index = pd.to_datetime(btc['Date'],infer_datetime_format =True,unit='s')
        
        (normalized_values ,bbp,moving_avarage,rsi_val,momentum,sma_cross) = indicators(data=btc)
        
        norm_val          = normalized_values.copy()
                
        states            = pd.concat([normalized_values,bbp,moving_avarage,rsi_val,momentum,sma_cross],axis=1).apply(lambda x : x.fillna(0)).iloc[13:,:]
                
        self.prices       = normalized_values.copy()

        state_size        = 6    # Tamanio del vector de estados
        action_size       = 5
        
        max_iter          = 10    # Iteraciones Maximas para el  aprendizaje
        actions_df        = pd.DataFrame(index=states.index,data=[0]*len(states))
        iter_num          = 0
        converged         = False
        
        dates = pd.date_range(sd,ed)
        
        #self.prices_all   = btc['Weighted_Price'] #ut.get_data([symbol], dates)[symbol]
        self.prices_all   = btc['Close'] #ut.get_data([symbol], dates)[symbol] 
                
        agent = DQNAgent(state_size, action_size)
        
        batch_size = 64  #64/32
        
        comienzo = time.time()

        for e in range(max_iter):
            
            print("Va por la iteracion ", e)
            
            holdings_actions = 0
            syms             = [symbol]
            X                = np.array([states.iloc[0]])
            action           = 2

            
            for key,row in states.iloc[1:].iterrows():
                
                                
                (reward,holdings_actions) = self.calculate_reward( holdings_actions_1 = holdings_actions , action = action ,key= key)
                
                next_state = np.array([row])
                
                agent.remember(X, action, reward, next_state)
                
                
                if len(agent.memory) > batch_size:
                    agent.replay(batch_size)
                    X = np.array([row])
                
                
                holdings_actions_1 = holdings_actions
                
                action = agent.act(X)
                
                actions_df.loc[key,iter_num] = holdings_actions

            iter_num += 1
            
            
            converged=False    
        print(time.time()-comienzo)
        previous_days = 13
        trades = pd.DataFrame(data = actions_df.iloc[:,-1],index = actions_df.index).diff().shift(-1).fillna(0)
        
        trades.sort_index(axis =0 , inplace=True)
        trades.iloc[-1] = -1*trades.iloc[:-1].sum()
        
        trades.columns = ['Shares']
        
        trades['Symbol'] = symbol
        trades['Order']  = trades['Shares'].to_frame().applymap(lambda x : {-2:'SELL',-1.5:'SELL',-1:'SELL',-0.5:'SELL',0:0,0.5:"BUY",1:"BUY",1.5:"BUY",2:"BUY"}[x])
        trades['Shares'] = trades['Shares'].abs()
        trades['Date']   = trades.index
        compute_portvals(trades)
        pdb.set_trace()
        self.agent = agent
        
        
    def testPolicy(self, symbol = "IBM", \
        sd=dt.datetime(2009,1,1), \
        ed=dt.datetime(2010,1,1), \
        sv = 10000):
        
        btc = pd.read_csv("COINBASE_FILTERED.CSV")
        
        size= int(len(btc)*0.005)
        btc = btc.iloc[-size:]
        
        btc[btc.columns.values] = btc[btc.columns.values].ffill()
        
        btc['TR'] =  0 
        
        a = btc['High']-btc['Low']
        b = btc['Low']-btc['Close'].shift(-1)
        c = btc['High']-btc['Close'].shift(-1)
        
        btc['TR'] = pd.concat([a,b,c],axis=1).max(axis=1)
        btc['ATR'] = btc['TR'].ewm(span = 10).mean()
        
        btc['Delta'] = btc['Close'] - btc['Open']
        
        btc['to_predict'] = btc['Delta'].apply(lambda x : 1 if(x>0) else 0)
        
        btc.index = pd.to_datetime(btc['Timestamp'],infer_datetime_format =True,unit='s')
        
        #self.prices_all   = btc['Weighted_Price'] #ut.get_data([symbol], dates)[symbol]
        self.prices_all   = btc['Close'] #ut.get_data([symbol], dates)[symbol]
        
        (normalized_values ,bbp,moving_avarage,rsi_val,momentum,sma_cross) = indicators(data=btc)
                
                
        norm_val          = normalized_values.copy()
        
        states            = pd.concat([normalized_values,bbp,moving_avarage,rsi_val,momentum,sma_cross],axis=1).apply(lambda x : x.fillna(0)).iloc[13:,:]
        
        actions_df        = pd.DataFrame(index=states.index,data=[0]*len(states))
        
        holdings          = pd.DataFrame(data = [0]*len(states) , index= states.index)
        
        for key,state in states.iterrows():
            action = self.agent.act(np.array([state]))

            holdings.loc[key] = {0:-1,1:-0.5,2:0 ,3:0.5 , 4:1}[action]

        pdb.set_trace()
        
        
        previous_days = 13
        trades = pd.DataFrame(data = actions_df.iloc[:,-1],index = actions_df.index).diff().shift(-1).fillna(0)
        
        trades.sort_index(axis =0 , inplace=True)
        trades.iloc[-1] = -1*trades.iloc[:-1].sum()
        
        
        
        trades.columns = ['Shares']
        

        trades['Symbol'] = symbol
        trades['Order']  = trades['Shares'].to_frame().applymap(lambda x : {-2:'SELL',-1.5:'SELL',-1:'SELL',-0.5:'SELL',0:0,0.5:"BUY",1:"BUY",1.5:"BUY",2:"BUY"}[x])
        trades['Shares'] = trades['Shares'].abs()
        trades['Date']   = trades.index
        self.agent       = agent
        
        if self.verbose: print(prices_all)
        return trades

if __name__=="__main__":
       
    nuevo = StrategyLearner(commission=9.95)
    nuevo.addEvidence(symbol="JPM")
    nuevo.testPolicy(symbol="JPM",sv=1000000)
    print ("One does not simply think up a strategy")
