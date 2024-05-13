from directinal_change import directional_change,get_extremes
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

df = pd.read_csv('BTCUSDT3600.csv')
extremes = get_extremes(df,0.02)
print(extremes.head(10))
# plt.plot(extremes[:10]['ext_i'].tolist(),extremes[:10]['ext_p'].tolist())
# plt.plot([0,1,2,3],[2,3,4,5])
# fig = px.line(x=extremes[:10]['ext_i'].tolist(),y=extremes[:10]['ext_p'].tolist())
fig= px.line(extremes,x='ext_i',y='ext_p')
fig.show()

