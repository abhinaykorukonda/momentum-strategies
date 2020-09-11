import pandas as pd
from strategies import SingleStockStrategy, QuantileStrategy, OptimizationStrategy
from os.path import join, dirname
import numpy as np


def main():

    processed_data_path = join(
        dirname(__file__), '..', 'data', 'processed')

    print("Loading processed files")
    # Load the signal file
    signal_df = pd.read_csv(join(processed_data_path, 'signal.csv'),
                            index_col=0)
    # Load stock monthly returns
    monthly_returns = pd.read_csv(join(processed_data_path, 'monthly_returns.csv'),
                                  index_col=0)
    # Load stock volatilties
    stock_vols = pd.read_csv(join(processed_data_path, 'monthly_vol.csv'),
                             index_col=0)
    # Load tickers with sectors
    sectors_sr = pd.read_csv(join(processed_data_path,  'sectors.csv'))

    sector_groupings = sectors_sr.reset_index().groupby(
        'Sector')['Symbol'].apply(lambda x: list(np.unique(x))).to_list()
    
    # Load covariances
    covariances_df = pd.read_csv(
        join(processed_data_path, 'monthly_covariances.csv'), index_col=[0, 1])

    """ Strategy 1 : Single stock strrategy """

    # Strategy 1.1 : Single stock strategy with no restrictions on risk tolerance
    print("Running single stock strategy")
    strat = SingleStockStrategy(signal_df)
    strat.compute_weights()
    strat_weights = strat.get_weights()
    strat_returns = strat.compute_returns(monthly_returns)
    print("Strategy complete")
    print("Dumped portfolio weights and returns to results folder")

    # Strategy 1.2 : Single stock strategy with a restriction on risk tolerance
    # by using a limit of 25% annualized volaitlity on the stock
    print("Running single stock strategy with a annualized volatility constraint of 25%")
    strat = SingleStockStrategy(signal_df, stock_vols,
                                risk_constraint=True, vol_tolerance=0.25)
    strat.compute_weights()
    strat_weights = strat.get_weights()
    strat_returns = strat.compute_returns(monthly_returns)
    print("Strategy complete")
    print("Dumped portfolio weights and returns to results folder")


    """Strategy 2: no constraint on single stock, but a long only constraint"""

    # Strategy 2.1 Trade top 5th quintile of the stocks based on the signal value
    print("Running top quintile strategy with long-only constraint")
    strat = QuantileStrategy(signal_df, quantiles=5)
    strat.compute_weights()
    strat_weights = strat.get_weights()
    strat_returns = strat.compute_returns(monthly_returns)
    print("Strategy complete")
    print("Dumped portfolio weights and returns to results folder")


    """Strategy 2: no constraint on single stock, but a long only constraint"""

    # Strategy 2.1 Trade top 5th quintile of the stocks based on the signal value
    print("Running top quintile strategy with long-only and sector neutrality constraints")

    strat = QuantileStrategy(
        signal_df, quantiles=5, sector_neutral=True, sector_groupings=sector_groupings)
    strat.compute_weights()
    strat_weights = strat.get_weights()
    strat_returns = strat.compute_returns(monthly_returns)
    print("Strategy complete")
    print("Dumped portfolio weights and returns to results folder")


    """Strategy 2: no constraint on single stock, but a long only constraint"""

    # Strategy 2.1 Trade top 5th quintile of the stocks based on the signal value
    print("Running optimization strategy with a target volatility of 15%")
    strat = OptimizationStrategy(signal_df, covariances_df, 0.15, True)
    strat.compute_weights()
    strat_weights = strat.get_weights()
    strat_returns = strat.compute_returns(monthly_returns)
    print("Strategy complete")
    print("Dumped portfolio weights and returns to results folder")


if __name__ == "__main__":

    main()
