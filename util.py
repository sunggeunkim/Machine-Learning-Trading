from sklearn.metrics import r2_score, mean_absolute_error, f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

# All rights reserved by Hye Joo Han
# reference: https://github.com/math470/Springboard_Capstone_Project_1
# A function that draw a feature importance plot
def plot_feature_importances(model_name, importances, feature_names, num_features=None,
                             fig_size=None, ax=None):
    features_importances = pd.DataFrame(sorted(zip(feature_names, importances),
                                           key = lambda tup: tup[1], reverse=True),
                                   columns=['features','importances'])
    data = features_importances[:num_features]
    data.plot('features','importances', kind='barh', ax=ax,
              color='blue', figsize=fig_size, legend = None)
    plt.gca().invert_yaxis() # reverse the order of features
    plt.ylabel('feature importances')
    if num_features:
        plt.title(model_name + '\nTop '+str(num_features))
    else:
        plt.title(model_name + '\nAll Features')

def rolling_mean(df, length):
    return df.rolling(window=length, center=False).mean()

def calc_features_seq(data, look_back=7, fee=0):
    AdjClose = data['Adj Close']
    data['Change'] = (data['Close']-data['Open']).map(lambda x: 1 if x>fee else 0)
    for i in range(look_back, 0, -1):
        data['Change' + str(i) + 'd'] = 100 * data['Close'].shift(i)/data['Open'].shift(i) - 100.0
    return data

def calc_features_shift(data, look_back=7, fee=0):
    AdjClose = data['Adj Close']
    #data['OverNightReturn'] = data['Adj Close'].shift(1)/data['Open'] - 1.0
    data['Change'] = (data['Close']-data['Open']).map(lambda x: 1 if x>fee else 0)
    for i in range(1, look_back+1):
        if i < 7 or i % 20 == 0:
            data['Return' + str(i) + 'd'] = data['Adj Close'].shift(i)/data['Open'] - 1.0
        if i == 120:
            data['dev_'+str(i)+'d'] = deviation_from_mean(AdjClose, i).shift(1)
    return data

# calculate features
def calc_features_new(data, look_back=1, fee=0):
    days = 120
    AdjClose = data['Adj Close']
    Volume = data['Volume']
    data['Change'] = (data['Close']-data['Open']).map(lambda x: 1 if x>fee else 0)
    data['OverNightReturn'] = data['Open']/data['Close'].shift(1) - 1.0
    for i in range(1,look_back+1):
        #if i < 10 or i % 20 == 0:
            #data['Change_'+str(i)+'d'] = data['Change'].shift(i)
            #data['daily_return_'+str(i)+'d'] = calc_trailing_daily_return(data['Adj Close'], i) 
        if i == 20 or i == 120 or i == 240:
            data['momentum_'+str(i)+'d'] = calc_momentum(AdjClose, i).shift(1)
            data['dev_'+str(i)+'d'] = deviation_from_mean(AdjClose, i).shift(1)
            data['dev_vol_'+str(i)+'d'] = deviation_from_mean(Volume, i).shift(1)
    return data

# remove data with same open and close prices
def remove_data_with_same_open_close(data):
    return data[abs(data['Open']-data['Close'])>1e-10]

# calculate devation from rolling mean
def deviation_from_mean(df, length):
    ave = rolling_mean(df, length)
    std = df.rolling(window=length, center=False).std()
    return (df - ave)/std

def rolling_mean(df, length):
    return df.rolling(window=length, center=False).mean()

def calc_trailing_daily_return(data, i):
    return data.shift(1) / data.shift(i+1) - 1.0

# calculate momentum
def calc_momentum(df, window):
    momentum = df.copy()
    momentum[window:] = (df[window:]/df[:-1*window].values) - 1
    momentum.ix[:window] = 0
    return momentum

# calculate daily return
def calc_daily_return(data):
    data = data/data.shift(1) - 1.0
    return data

# calculate accuracy
def accuracy(y_test, y_pred, type="mean_absolute_error"):
    if type == "r2_score":
        acc = r2_score(y_test, y_pred)
    elif type == "mean_absolute_error":
        acc = mean_absolute_error(y_test, y_pred)
    elif type == "accuracy_score":
        acc = accuracy_score(y_test, y_pred) * 100
    elif type == "f1_score":
        acc = f1_score(y_test, y_pred)
    return acc

def gain_plot_from_test_data(data_test, y_pred, fee=0, model_name = ""):
    nstock = 100 #number of stocks to trade
    price = data_test.iloc[0]['Open'] * nstock
    asset = np.zeros(len(data_test))
    asset[0] = price
    
    #calculate ROI based on prediction
    for i in range(len(data_test)):
        if y_pred[i]>0.5:
            #buy n stocks
            price += -data_test.iloc[i]['Open'] * nstock + data_test.iloc[i]['Close'] * nstock - fee
        else:
            #sell n stocks
            price += data_test.iloc[i]['Open'] * nstock - data_test.iloc[i]['Close'] * nstock - fee
        asset[i] = price
    start_asset = data_test.iloc[0]['Open'] * nstock
    ref = start_asset + data_test['Close'] * nstock - data_test.iloc[0]['Open']*nstock
    
    #plot ROI
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(ref / start_asset, label='buy-hold')
    ax.plot(pd.DataFrame(asset, index=data_test.index) / start_asset, label=model_name)
    years = mdates.YearLocator(10)   # every year
    months = mdates.MonthLocator()  # every month
    yearsFmt = mdates.DateFormatter('%Y')
    ax.xaxis.set_major_locator(years)
    ax.xaxis.set_major_formatter(yearsFmt)
    ax.xaxis.set_minor_locator(months)
    ax.legend(loc="upper left", bbox_to_anchor=[0, 1],
           ncol=1, shadow=True, fancybox=True)
    ax.set_xlabel('Date')
    ax.set_ylabel('Return on investment')
    plt.tight_layout()
    plt.show()
    roi = asset[-1]/start_asset
    return roi
