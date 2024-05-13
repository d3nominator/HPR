#This script constructs retracement ratios for different values of sigma parameter
# and plots their density
import pandas as pd
import numpy as np
import scipy
import math
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from Directional_Change.directinal_change import get_extremes,directional_change

data = pd.read_csv('BTCUSDT3600.csv')
data['date'] = data['date'].astype('datetime64[s]')
data = data.set_index('date')
plt.style.use('dark_background')

for sigma in [0.04]:
    extremes = get_extremes(data,sigma)

    #Find the segment heights , retracements ratios
    extremes['seg_height'] = (extremes['ext_p'] - extremes['ext_p'].shift(1)).abs()
    extremes['retrace_ratio'] = extremes['seg_height'] / extremes['seg_height'].shift(1)
    extremes['log_retrace_ratio'] = np.log(extremes['retrace_ratio'])
    # fig = px.histogram(extremes,x='log_retrace_ratio')
    # fig.show()

    #Find the kernel of log retrace ratios
    kernel = scipy.stats.gaussian_kde(extremes['log_retrace_ratio'].dropna(),bw_method=0.01)
    # plt.plot(np.arange(-3,3,0.01),kernel(np.arange(-3,3,0.01)))
    # plt.show()
    # sns.kdeplot(extremes['log_retrace_ratio'])
    # plt.show()

    retrace_range = np.arange(-3,3,0.001)
    retrace_pdf = kernel(retrace_range)
    retrace_pdf =  pd.Series(retrace_pdf,index=np.exp(retrace_range))
    retrace_pdf.plot(color='orange',label='Retrace PDF')
    plt.axvline(0.618,label='0.618',color='blue')
    plt.axvline(1.618,label='1.618',color='red')
    plt.title("Retracement density (sigma=" + str(sigma) + ")")
    plt.legend()
    plt.show()