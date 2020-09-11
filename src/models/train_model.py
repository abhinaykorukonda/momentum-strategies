from scipy.optimize import minimize
import pandas as pd
import numpy as np
from os.path import join, dirname


processed_data_path = join(dirname(__file__), '..', '..', 'data', 'processed')

# monthly_returns = pd.read_csv(join(processed_data_path, 'monthly_returns.csv'))

signal = pd.read_csv(join(processed_data_path, 'signal.csv'), index_col=0)
sectors = pd.read_csv(join(processed_data_path, 'sectors.csv'), index_col = 0)

benchmark_returns = pd.read_csv(
    join(processed_data_path, 'benchmark_returns.csv'), index_col=0).iloc[:, 0]
benchmark_vol = pd.read_csv(
    join(processed_data_path, 'benchmark_vol.csv'), index_col=0).iloc[:, 0]


covariances_df = pd.read_csv(
    join(processed_data_path, 'monthly_covariances.csv'), index_col=[0, 1])
vol_df = pd.read_csv(join(processed_data_path, 'monthly_vol.csv'), index_col=0)


""" 
Strategy 1

Pick only one stock at a time. This would simply mean pick the stock with the highest return 
while still under the boundaries of the risk tolerance defined by the benchmark
"""

weights = {}

for date, row in signal.iterrows():

    row_mask = vol_df.loc[vol_df.index == date, :] <= benchmark_vol[date]
    if np.any(row_mask):
        traded_stock = row.loc[row_mask.T.iloc[:, 0].values].idxmax()
    else:
        traded_stock = row.idxmax()

    weights[date] = (row.index == traded_stock) * 1

weights_df = pd.DataFrame.from_dict(weights).T
weights_df.columns = signal.columns


single_strategy = weights_df


# Optimization without sector neutrality


monthend_dates = signal.index.values

weights = {}

for sample_date in monthend_dates[36:]:
    print(sample_date)

    covariance = covariances_df.xs(sample_date, level=0)
    alpha = signal.loc[signal.index == sample_date, :].T.iloc[:, 0]
    alpha = alpha.loc[~alpha.isnull()]
    covariance = covariance.loc[alpha.index, alpha.index]

    def portfolio_std(w, covariance):
        variance = (w.T @ covariance @ w) - 0.05

        return variance


    def weight_sum(w):
        return np.sum(w) - 1


    def portfolio_alpha(w, alpha):

        return np.sum(w * alpha) * -1


    w0 = np.ones(alpha.shape) / alpha.shape

    constraints = [{'type': 'eq', 'fun': lambda x: portfolio_std(x, covariance)},
                {'type': 'eq', 'fun': weight_sum}]


    bounds = [(0, 1)]*(w0.shape[0])

    ans = minimize(portfolio_alpha, w0, constraints=constraints,
                args=(alpha,), bounds=bounds)
    
    weights[sample_date] = dict(zip(alpha.index,ans.x))
    break

unconstrained_strategy = pd.DataFrame.from_dict(weights).T.fillna(0)



## Strategy 3 


sector_constraints = []

uniq_sectors = ['Industrials', 'Health Care', 'Information Technology',
       'Communication Services', 'Consumer Discretionary', 'Utilities',
       'Financials', 'Materials', 'Real Estate', 'Consumer Staples',
       'Energy']

monthend_dates = signal.index.values

weights = {}


for sample_date in monthend_dates[36:]:
    print(sample_date)

    covariance = covariances_df.xs(sample_date, level=0)
    alpha = signal.loc[signal.index == sample_date, :].T.iloc[:, 0]
    alpha = alpha.loc[~alpha.isnull()]
    covariance = covariance.loc[alpha.index, alpha.index]

    def portfolio_std(w, covariance):
        variance = (w.T @ covariance @ w) - 0.05

        return variance


    def weight_sum(w):
        return np.sum(w) - 1


    def portfolio_alpha(w, alpha):

        return np.sum(w * alpha) * -1


    w0 = np.ones(alpha.shape) / alpha.shape

    constraints = [{'type': 'eq', 'fun': lambda x: portfolio_std(x, covariance)},
                {'type': 'eq', 'fun': weight_sum}]

    alpha_sectors = sectors.loc[alpha.index,'Sector']


    for each_sector in uniq_sectors:
        single_sector_constraint = {'type' : 'eq','fun' : lambda x: x[alpha_sectors == each_sector] - 1/11 }
        
        constraints.append(single_sector_constraint)

    bounds = [(0, 1)]*(w0.shape[0])

    ans = minimize(portfolio_alpha, w0, constraints=constraints,
                args=(alpha,), bounds=bounds)
    
    weights[sample_date] = dict(zip(alpha.index,ans.x))

    break
