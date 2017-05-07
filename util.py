from sklearn.metrics import r2_score, mean_absolute_error, f1_score, accuracy_score

# calculate features
def calc_features_new(data, look_back=1, fee=0):
    days = 120
    AdjClose = data['Adj Close']
    Volume = data['Volume']
    data['Change'] = (data['Close']-data['Open']).map(lambda x: 1 if x>fee else 0)
    data['OverNightReturn'] = data['Open']/data['Close'].shift(1) - 1.0
    for i in range(1,look_back+1):
        if i < 10 or i % 20 == 0:
            data['Change_'+str(i)+'d'] = data['Change'].shift(i)
            data['daily_return_'+str(i)+'d'] = calc_trailing_daily_return(data['Adj Close'], i) 
        if i % 20 == 0:
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

