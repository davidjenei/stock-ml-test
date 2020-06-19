# %%
def get_feature_importance_data(X,y):
    
    train_samples = int(len(X) * 0.65)
 
    X_train = X.iloc[:train_samples]
    X_test = X.iloc[train_samples:-1]

    y_train = y.iloc[:train_samples]
    y_test = y.iloc[train_samples:-1]
  
    return (X_train, y_train), (X_test, y_test)
# %% 
# Download historical price
import yfinance as yf
import matplotlib.pyplot as plt
from ta import add_all_ta_features
from numpy import log10
import pandas_ta as ta

tickers = ["AAPL","MSFT","SPY","NDAQ"]
start = "2010-02-01"
end = "2020-06-01"

stock = yf.download(tickers, start="2010-02-01", end="2020-06-01") \
        .rename(columns={"Open": "open", "High": "high", "Low": "low", \
          "Close": "close", "Volume": "volume", "Adj Close": "adj_close"}) \
        .stack()

stock.index.set_names(['date', 'symbol'], inplace=True)

# %% 
# Download volatility and option statistics
from requests import get
import pandas as pd
import numpy as np
from pandas import read_json, DatetimeIndex
ids = ['put-call-ratio-volume','put-call-ratio-oi','call-breakeven', \
       'put-breakeven','option-breakeven']
type = '90-Day'

for id in ids:
  for ticker in tickers:
    url = "https://www.alphaquery.com/data/option-statistic-chart?ticker=" + \
           ticker + "&perType=" + type + "&identifier=" + id
    id_ = id.replace('-','_')
    tmp = read_json(get(url).content) \
          .rename(columns={"value": id_, "x": "date"})
    tmp['date'] = DatetimeIndex(tmp.date).normalize().tz_localize(None)
    tmp['symbol'] = ticker
    tmp.set_index('date',inplace=True)
    tmp.set_index('symbol',append=True,inplace=True)
    if id_ in stock: 
      stock.update(tmp) 
    else:
      stock[id_] = tmp

# %%
# Shift close price
outcomes = pd.DataFrame(index=stock.index)
# next day's opening change
outcomes['open_1'] = stock.groupby(level='symbol').open.shift(-1) \
/stock.groupby(level='symbol').close.shift(0)-1
outcomes['open_next'] = stock.groupby(level='symbol').open.shift(-1)
# next day's closing change
func_one_day_ahead = lambda x: x.pct_change(-1)
outcomes['close_1'] = stock.groupby(level='symbol').close \
.apply(func_one_day_ahead)
func_five_day_ahead = lambda x: x.pct_change(-5)
outcomes['close_5'] = stock.groupby(level='symbol').close \
.apply(func_five_day_ahead)

# Drop last line with no closing price
# outcomes.drop(outcomes.groupby('symbol').tail(1).index,inplace=True)

# %%
# Select features
from ta.momentum import wr, stoch, rsi, roc
from ta.trend import macd
from ta.volume import on_balance_volume

features = pd.DataFrame()

# each variable is evaluated as a potential splitting variable, which makes them
# robust to unimportant/irrelevant variables, because such variables that cannot
# discriminate between events/non-events will not be selected as the splitting 
# variable and hence will be very low on the var importance graph as well.

# Daily return
features['f01'] = stock.close/stock.open-1
features['f010'] = features.groupby(level='symbol').f01.shift()
# Open gap
features['f02'] = stock.open/stock.groupby(level='symbol').close.shift(1)-1

# Put call ratio change
features['f03'] = stock.put_call_ratio_volume.diff()
features['f04'] = stock.volume.apply(np.log)
features['f05'] = stock.groupby(level='symbol').close.apply(rsi)

func_stoch = lambda x: stoch(high=x.high, low=x.low, close=x.close)
features['f06'] = stock.groupby(level='symbol').apply(func_stoch).reset_index(drop=True,level=0)

# func_wr = lambda x: wr(high=x.high,low=x.low, close=x.close)
# features['f07'] = stock.groupby(level='symbol').apply(func_wr).reset_index(drop=True,level=0)

# features['f08'] = stock.groupby(level='symbol').close.apply(macd)

# func_ema_50 = lambda x: x.ewm(alpha=0.095).mean()
# features['f09'] = stock.close/ stock.close.groupby(level='symbol').apply(func_ema_50)-1

# features['f10'] = stock.groupby(level='symbol').close.apply(roc)
# Signing
features['f11'] = features['f01'].apply(np.sign)
# %% 
# XGBoost model
import xgboost as xgb
from xgboost import plot_importance
from sklearn.metrics import accuracy_score
import plotly.express as px

px.defaults.width = 600
px.defaults.height = 300

X_xgb = features.unstack()
y_xgb = outcomes.xs('AAPL',level='symbol')['close_1']
(X_train_FI, y_train_FI), (X_test_FI, y_test_FI) = \
    get_feature_importance_data(X_xgb,y_xgb)

regressor = xgb.XGBRegressor(
    gamma=0, 
    max_depth=18, # size of decision trees
    n_estimators=113, # number of trees
    base_score=0.7,
    min_child_weight=7,
    colsample_bytree=1,
    learning_rate=0.05)
xgbModel = regressor.fit(X_train_FI,y_train_FI, \
  eval_set = [(X_train_FI, y_train_FI), (X_test_FI, y_test_FI)], \
  verbose=False)
eval_result = regressor.evals_result()
training_rounds = range(len(eval_result['validation_0']['rmse']))

df_validation = pd.DataFrame({'training': eval_result['validation_0']['rmse'], 'validation': eval_result['validation_1']['rmse']})
px.scatter(df_validation)

# %% 
# Feature importance type: gain, weight, cover
import plotly.express as px

feature_score = regressor.get_booster().get_score(importance_type='gain')
keys = list(feature_score.keys())
values = list(feature_score.values())

df_imp = pd.DataFrame({'col': keys,'imp': values}).sort_values(by = "imp", ascending=False)
px.bar(df_imp,x='col',y='imp')

# %%
# Feature orthogonality
# Dendrogram: allocate objects to clusters
from scipy.cluster import hierarchy
from scipy.spatial import distance
import seaborn as sns

corr_matrix = features.unstack().corr()
correlations_array = np.asarray(corr_matrix)

linkage = hierarchy.linkage(distance.pdist(correlations_array), \
                            method='average')

g = sns.clustermap(corr_matrix,row_linkage=linkage,col_linkage=linkage,\
                   row_cluster=True,col_cluster=True,figsize=(10,10), \
                   cmap='Greens')
plt.setp(g.ax_heatmap.yaxis.get_majorticklabels(), rotation=0)
plt.show()

# %%
# Evaluate predictions
def calc_prediction(y_pred, y_true):
  df = pd.DataFrame(y_true)
  df['y_pred'] = y_pred
  df['sign_pred'] = df.y_pred.apply(np.sign)
  df['sign_true'] = df.y_true.apply(np.sign)
  df['is_correct'] = 0
  df.loc[df.sign_pred * df.sign_true > 0 ,'is_correct'] = 1 # only registers 1 when prediction was made AND it was correct
  df['is_incorrect'] = 0
  df.loc[df.sign_pred * df.sign_true < 0,'is_incorrect'] = 1 # only registers 1 when prediction was made AND it was wrong
  df['is_predicted'] = df.is_correct + df.is_incorrect
  df['result'] = df.sign_pred * df.y_true
 
  return df

y_pred = regressor.predict(X_test_FI).astype(np.float64)
y_true = y_test_FI.rename('y_true')

df_pred = calc_prediction(y_pred, y_true)
px.line(df_pred[['y_true','y_pred']])

# %%
# Calculate scorecard
def calc_scorecard(df):
    scorecard = pd.Series(dtype='float64')
    # building block metrics
    # accurancy: directionally correct vs. incorrect.
    scorecard.loc['accuracy'] = df.is_correct.sum()*1. / (df.is_predicted.sum()*1.)*100
    # edge: expected profit per time period
    scorecard.loc['edge'] = df.result.mean()
    scorecard.loc['noise'] = df.y_pred.diff().abs().mean()
    
    return scorecard 

df_score = calc_scorecard(df_pred)
print(df_score)

# %%
from scipy.stats import uniform
from sklearn.model_selection import RandomizedSearchCV
param_dist = {"learning_rate": uniform(0, 1),
                 "gamma": uniform(0, 5),
                 "max_depth": range(1,50),
                "n_estimators": range(1,300),
                "min_child_weight": range(1,10)}

#rs = RandomizedSearchCV(regressor, param_distributions=param_dist, 
                          scoring='neg_mean_squared_error', n_iter=25)

# # Run random search for 25 iterations
# rs.fit(X_test_FI.head(-1), y_test_FI.head(-1))
# print(rs.best_score_)
# rs.best_params_

# %%
