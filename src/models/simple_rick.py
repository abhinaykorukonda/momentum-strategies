from scipy.optimize import minimize
import pandas as pd
import numpy as np
from os.path import join, dirname


processed_data_path = join(dirname(__file__), '..', '..', 'data', 'processed')

# monthly_returns = pd.read_csv(join(processed_data_path, 'monthly_returns.csv'))

signal = pd.read_csv(join(processed_data_path, 'signal.csv'), index_col=0)
sectors = pd.read_csv(join(processed_data_path, 'sectors.csv'), index_col=0)

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


# Top quintile strategy


weights = signal.apply(lambda x: x >= np.quantile(x.dropna(), 0.8), axis=1) * 1


weights = weights.divide(weights.sum(axis=1), axis=0)
