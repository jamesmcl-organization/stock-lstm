import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates

# Indexes to run through and compared
'''
HUI:    Gold
XAU:    Silver
DXY:    USD
CL:     Oil
DJI:    Dow
SPY:    S&P 500 Index
NDAQ:   Nasdaq
'''

index_tickers = [ 'HUI', 'XAU', 'DXY', 'CL', 'DJI', 'SPY', 'NDAQ' ]

indicators = [ 'dma', 'volume_delta', 'close_12_ema', 'close_26_ema', 'macd', 'macd_9_ema', 'macds', 'macdh' ]

#df = pd.read_csv (r'/home/ubuntu/stock_nn/export_stock_data.csv', parse_dates=[ 'date' ], squeeze=True)
#headers = pd.read_csv (r'/Users/jamesm/Desktop/Data_Science/stock_lstm/export_files/headers.csv')
headers = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/headers.csv')
#df = pd.read_csv (r'/Users/jamesm/Desktop/Data_Science/stock_lstm/export_files/stock_history.csv', header=None, names=list(headers))
df = pd.read_csv (r'/home/ubuntu/stock_lstm/export_files/stock_history.csv', header=None, names=list(headers))

# Extract the index tickers and AAPL stock
subset = df [ (df [ 'ticker' ].isin (index_tickers)) | (df [ 'ticker' ] == 'AAPL') ]
subset.index.name = 'date'

aapl = subset [ subset [ 'ticker' ] == 'AAPL' ]
symbols = aapl [ 'ticker' ].unique ()



aapl.to_csv (r'/home/ubuntu/stock_lstm/export_files/aapl.csv', index=True, header=True)
#aapl.to_csv (r'/Users/jamesm/Desktop/Data_Science/stock_lstm/export_files/aapl.csv', index=True, header=True)

ticker = str (aapl [ 'ticker' ].unique () [ 1:-1 ])

# Calculate Previous day's Volume Change %

fig, axes = plt.subplots (nrows=4, figsize=(20, 20))
fig.suptitle (ticker)

for j, ax in enumerate (axes):
    axes [ j ].xaxis.set_major_locator (mdates.MonthLocator (interval=6))  # to display ticks every 3 months
    axes [ j ].xaxis.set_major_formatter (mdates.DateFormatter ('%Y-%m'))  # to set how dates are displayed
    print (axes [ j ])

# Plot the Moving Averages
axes [ 0 ].set_title ('Close and MAs')
axes [ 0 ].plot (aapl.index, aapl [ [ 'close', 'ma20', 'ma50', 'ma200' ] ], linewidth=3)

# Plot the MACD-H Chart
axes [ 1 ].set_title ('MACDH')
axes [ 1 ].bar (aapl.index, aapl [ 'macdh' ], lw=10)
axes [ 1 ].plot (aapl.index, aapl [ 'macds' ], color='red')
axes [ 1 ].plot (aapl.index, aapl [ 'macd' ], color='black')
axes [ 1 ].set_ylim ([ -8, 8 ])

# Plot the % Change from previous close
axes [ 2 ].set_title ('% Change in Close')
axes [ 2 ].plot (aapl.index, aapl [ 'prev_close_ch' ], color='red')

axes [ 3 ].set_title ('Daily Volume')
axes [ 3 ].bar (aapl.index, aapl [ 'volume' ], color='blue')
# plt.legend()
plt.show ()

# Next we will look at the monthly % change of the stock over time, by month
df_hm = aapl.copy () [ [ 'close', 'ticker' ] ]
df_hm.reset_index (inplace=True)

df_hm [ 'year' ] = list (map (lambda x: x.year, df_hm [ 'date' ]))
df_hm [ 'month' ] = list (map (lambda x: x.month, df_hm [ 'date' ]))
df_hm [ 'day' ] = list (map (lambda x: x.day, df_hm [ 'date' ]))

# Capture the close at the beginning of each month (first()) then calculate the monthly returns (% change)
# and unstack by month - then plot the resultant heatmap
monthly_returns = df_hm.groupby ([ 'year', 'month' ]).first () [ 'close' ].pct_change ().unstack ('month')

sns.heatmap (monthly_returns, linecolor='white', lw=1)
plt.show ()

# Next do a correlation plot of the % change in the stock and index close prices
# In other words, when one moves hwo do the others react
sns.set(font_scale=3) # font size 2

fig = plt.figure(figsize = (20,20))
ax = plt.axes()

corr = subset [ [ 'ticker', 'close' ] ]
sns.heatmap(corr.pivot(index= corr.index , columns='ticker')['close'].corr(), cmap='coolwarm', annot=True, ax=ax, linewidths=3, linecolor="k")
ax.set_title('Correlation between the Index closes and AAPL')
plt.show()

#Now look at the correlations in 2020 only
fig = plt.figure(figsize = (20,20))
ax = plt.axes()

corr2020 = subset[subset.index.year == 2020] [ [ 'ticker', 'close' ] ]
sns.heatmap(corr2020.pivot(index= corr2020.index , columns='ticker')['close'].corr(), cmap='coolwarm', annot=True, ax=ax, linewidths=3, linecolor="k")
ax.set_title('Correlation between the 2020 Index closes and AAPL')
plt.show()

sns.set(font_scale=1) # font size 2