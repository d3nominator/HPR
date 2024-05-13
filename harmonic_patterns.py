from dataclasses import dataclass
from typing import Union
import pandas as pd
import math
import numpy as np  
from Directional_Change.directinal_change import get_extremes
import matplotlib.pyplot as plt
import mplfinance as mpf

@dataclass
class XABCD:
    XA_AB: Union[float,list,None]
    AB_BC: Union[float,list,None]
    BC_CD: Union[float,list,None]
    XA_AD: Union[float,list,None]
    name: str

#define patterns
GARTELY = XABCD(0.618,[0.382,0.886],[1.13,1.618],0.786,"Gartely")
BAT = XABCD([0.382,0.50],[0.382,0.886],[1.618,2.618],0.886,"Bat")
BUTTERFLY = XABCD(0.786,[0.382,0.886],[1.68,2.24],[1.27,1.41],"Butterfly")
CRAB = XABCD([0.382,0.618],[0.382,0.886],[2.618,3.618],1.618,"Crab")
DEEP_CRAB = XABCD(0.886,[0.382,0.886],[2.0,3.618],1.618,"Deep Crab")
CYPHER = XABCD([0.382,0.618],[1.13,1.41],[1.27,2.00],0.786,"Cypher")
SHARK = XABCD(None,[1.13,1.618],[1.618,2.24],[0.886,1.13],"shark")

ALL_PATTERNS = [GARTELY,BAT,BUTTERFLY,CRAB,DEEP_CRAB,CYPHER,SHARK]

@dataclass
class XABCDFound:
    X: int
    A: int
    B: int
    C: int
    D: int
    error: float
    name: str
    bull: bool

def plot_pattern(ohlc: pd.DataFrame, pat: XABCDFound, pad=3):
    idx = ohlc.index
    data = ohlc.iloc[pat.X - pad : pat.D + 1 + pad]
    plt.style.use('dark_background')
    fig = plt.gcf()
    ax = fig.gca()

    if pat.bull:
        s1 = ohlc['low'].to_numpy()
        s2 = ohlc['high'].to_numpy()
    else:
        s2 = ohlc['low'].to_numpy()
        s1 = ohlc['high'].to_numpy()
    
    l0 = [(idx[pat.X], s1[pat.X]), (idx[pat.A], s2[pat.A])]
    l1 = [(idx[pat.A], s2[pat.A]), (idx[pat.B], s1[pat.B])]
    l2 = [(idx[pat.B], s1[pat.B]), (idx[pat.C], s2[pat.C])]
    l3 = [(idx[pat.C], s2[pat.C]), (idx[pat.D], s1[pat.D])]

    # Connecting lines
    l4 = [(idx[pat.A], s2[pat.A]), (idx[pat.C], s2[pat.C])]
    l5 = [(idx[pat.B], s1[pat.B]), (idx[pat.D], s1[pat.D])]
    l6 = [(idx[pat.X], s1[pat.X]), (idx[pat.B], s1[pat.B])]
    l7 = [(idx[pat.X], s1[pat.X]), (idx[pat.D], s1[pat.D])]

    mpf.plot(
        data, 
        alines=dict(alines=[l0, l1, l2, l3, l4, l5, l6, l7 ], colors=['w', 'w', 'w', 'w', 'b', 'b', 'b', 'b']),
        type='candle', style='charles', ax=ax
    )

    # Text
    xa_ab =  abs(s2[pat.A] - s1[pat.B]) / abs(s1[pat.X] - s2[pat.A])
    ab_bc =  abs(s1[pat.B] - s2[pat.C]) / abs(s2[pat.A] - s1[pat.B])
    bc_cd =  abs(s2[pat.C] - s1[pat.D]) / abs(s1[pat.B] - s2[pat.C])
    xa_ad =  abs(s2[pat.A] - s1[pat.D]) / abs(s1[pat.X] - s2[pat.A])
    ax.text(int((pat.X + pat.B) / 2) - pat.X + pad, (s1[pat.X] + s1[pat.B]) / 2 , str(round(xa_ab, 3)), color='orange', fontsize='x-large')
    ax.text(int((pat.A + pat.C) / 2) - pat.X + pad, (s2[pat.A] + s2[pat.C]) / 2 , str(round(ab_bc, 3)), color='orange', fontsize='x-large')
    ax.text(int((pat.B + pat.D) / 2) - pat.X + pad, (s1[pat.B] + s1[pat.D]) / 2 , str(round(bc_cd, 3)), color='orange', fontsize='x-large')
    ax.text(int((pat.X + pat.D) / 2) - pat.X + pad, (s1[pat.X] + s1[pat.D]) / 2 , str(round(xa_ad, 3)), color='orange', fontsize='x-large')
    
    desc_string = pat.name
    desc_string += "\nError: " + str(round(pat.error , 5))
    if pat.bull:
        plt_price = data['high'].max() - 0.05 * (data['high'].max() - data['low'].min())
    else:
        plt_price = data['low'].min() + 0.05 * (data['high'].max() - data['low'].min())
    ax.text(0, plt_price , desc_string, color='yellow', fontsize='x-large')
    plt.show()



def get_error(actual_ratio: float,pattern_ratio = Union[float,list,None]):
    if pattern_ratio is None:#No requirement (Shark)
        return 0.0
    log_actual = math.log(actual_ratio)

    if isinstance(pattern_ratio,list):
        log_pat0 = math.log(pattern_ratio[0])
        log_pat1 = math.log(pattern_ratio[1])
        assert(log_pat1>log_pat0)

        if log_pat0 <= log_actual <= log_pat1:
            return 0.0
        # else:
        #     return 1e20

        err = min(abs(log_actual-log_pat0),abs(log_actual-log_pat1))
        range_mult = 2.0
        err *= range_mult
        return err
    elif isinstance(pattern_ratio,float):
        err = abs(log_actual - math.log(pattern_ratio))
        return err
    else:
        return TypeError("invalid pattern ration type")
    

def find_xabcd(ohlc: pd.DataFrame,extremes: pd.DataFrame,err_thres: float = 0.2):
    extremes['seg_height'] = (extremes['ext_p'] - extremes['ext_p'].shift(1)).abs()
    extremes['retrace_ratio'] = extremes['seg_height']/ extremes['seg_height'].shift(1)

    output = {}
    for pat in ALL_PATTERNS:
        pat_data = {}
        pat_data['bull_signal'] = np.zeros(len(ohlc))
        pat_data['bull_patterns'] = []
        pat_data['bear_signal'] = np.zeros(len(ohlc))
        pat_data['bear_pat'] = []
        output[pat.name] = pat_data

        first_conf = extremes.index[0]
        extreme_i = 0

        entry_taken = 0 # 0 zero for no position, 1 for bull position, -1 for bear position
        pattern_used = None
        for i in range(first_conf,len(ohlc)):
            if extremes.index[extreme_i+1] == i:
                entry_taken = 0
                extreme_i+=1
            
            if entry_taken != 0:
                if entry_taken == 1:
                    output[pattern_used]['bull_signal'][i] = 1
                else:
                    output[pattern_used]['brear_signal'][i] = -1
                continue
            if extreme_i + 1 >= len(extremes):
                break
            if extreme_i < 3:
                continue
            ext_type = extremes.iloc[extreme_i]['type']
            last_conf_i = extremes.index[extreme_i]

            if extremes.iloc[extreme_i]['type'] > 0.0:
                #Last was top meaning we are in down leg currently
                #We are checking for a bull pattern
                D_price = ohlc.iloc[i]['low']
                #Check that the current low is lowest since last two tops
                if ohlc.iloc[last_conf_i:i]['low'].min() < D_price:
                    continue
            else:
                #Last extreme was a bottom, meaning we are in the leg up currently
                #we are checking for bear pattern
                D_price = ohlc.iloc[i]['high']
                if ohlc.iloc[last_conf_i:i]['high'].max() > D_price:
                    continue
            
            #D_price set,get_ratios
            dc_retrace = abs(D_price - extremes.iloc[extreme_i]['ext_p'])/extremes.iloc[extreme_i]['seg_height']
            xa_ad_retrace = abs(D_price - extremes.iloc[extreme_i - 2 ]['ext_p']) / extremes.iloc[extreme_i - 2]['seg_height']

            best_err = 1e30
            best_pat = None
            for pat in ALL_PATTERNS:
                err = 0.0
                err += get_error(extremes.iloc[extreme_i]['retrace_ratio'],pat.AB_BC)
                err += get_error(extremes.iloc[extreme_i-1]['retrace_ration'],pat.XA_AB)
                err += get_error(dc_retrace,pat.BC_CD)
                err += get_error(xa_ad_retrace,pat.XA_AD)
                if err < best_err:
                    best_err = err
                    best_pat = pat.name

            if best_err <= err_thres:
                pattern_data = XABCDFound(
                    int(extremes.iloc[extreme_i-3]['ext_i']),
                    int(extremes.iloc[extreme_i-2]['ext_i']),
                    int(extremes.iloc[extreme_i-1]['ext_i']),
                    int(extremes.iloc[extreme_i]['ext_i']),
                    i,
                    best_err,best_pat,True
                )


                pattern_used = best_pat
                if ext_type > 0.0:
                    entry_taken = 1
                    pattern_data.name = "Bull" + pattern_data.name
                    pattern_data.bull = True
                    output[pattern_used]['bull_signal'][i] = 1
                    output[pattern_used]['bull_patterns'].append(pattern_data)
                else:
                    entry_taken = -1
                    pattern_data.name = "Bear" + pattern_data.name
                    pattern_data.bull = False
                    output[pattern_used]['bear_signal'][i] = -1
                    output[pattern_used]['bear_patterns'].append(pattern_data)

    return output

if __name__ == '__main__':
    data = pd.read_csv('BTCUSDT3600.csv')
    data['date'] = data['date'].astype('datetime64[s]')
    data = data.set_index('date')


    extremes = get_extremes(data,0.02)
    output = find_xabcd(data,extremes,0.2)
    print(output)
    # for pat in output:
    #     plot_pattern(data,pat)

