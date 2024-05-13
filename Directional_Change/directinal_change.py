import numpy as np
import pandas as pd
# from typing import

def directional_change(close: np.array, high: np.array, low: np.array, sigma: float):
    
    #Below code is for the current index
    up_zig = True #Last extreme is bottom, next is a top
    tmp_max = high.iloc[0]
    tmp_min = low.iloc[0]
    tmp_max_i = 0
    tmp_min_i = 0

    tops = []
    bottoms = []

    for i in range(len(close)):
        if up_zig: #Last extreme is bottom
            if high.iloc[i] > tmp_max:
                #New High,update
                tmp_max = high.iloc[i]
                tmp_max_i = i
            elif close.iloc[i] < tmp_max - tmp_max * sigma:
                # Price has retraced by sigma %. Top confirmed record it
                #top[0] = confirmation index
                #top[1] = index of top
                #top[2] = price of top
                top = [i,tmp_max_i,tmp_max]
                tops.append(top)

                #Set up for next Bottom 
                up_zig = False
                tmp_min = low.iloc[i]
                tmp_min_i = i
        else: #Last extreme is top
            if low.iloc[i] < tmp_min:
                #New Low, Update
                tmp_min = low.iloc[i]
                tmp_min_i = i
            elif close.iloc[i] > tmp_min_i + tmp_min * sigma:
                #Price retraced by sigma %, record it 
                # bottom[0] = confirmation Index
                # bottom[1] = index of bottom
                # bottom[2] = price of bottom
                bottom = [i,tmp_min_i,tmp_min]
                bottoms.append(bottom)

                #setup for next top
                up_zig = True
                tmp_max = high.iloc[i]
                tmp_max_i = i
    return tops,bottoms


def get_extremes(ohlc: pd.DataFrame, sigma: float):
    tops,bottoms = directional_change(ohlc['close'],ohlc['high'],ohlc['low'],sigma)
    tops = pd.DataFrame(tops,columns=['conf_i','ext_i','ext_p'])
    bottoms = pd.DataFrame(bottoms,columns=['conf_i','ext_i','ext_p'])
    tops['type'] = 1
    bottoms['type'] = -1
    extremes = pd.concat([tops,bottoms])
    extremes.set_index('conf_i')
    extremes = extremes.sort_index()
    return extremes
# How to test the above code 


        


