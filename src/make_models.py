import pandas as pd
from strategies import SingleStockStrategy, QuantileStrategy, OptimizationStrategy
from os.path import join, dirname
import numpy as np


OUTPUT_PATH = join(dirname(__file__), '..', 'data', 'results')
INPUT_PATH = join(dirname(__file__), '..', 'data', 'processed')


def dump_results(strat, stock_returns, strategy_name):

    strat_weights = strat.get_weights()
    strat_returns = strat.compute_returns(stock_returns)
    strat_weights.to_csv(
        join(OUTPUT_PATH, 'portfolio_weights', strategy_name + '.csv'))
    strat_returns.to_csv(
        join(OUTPUT_PATH, 'portfolio_returns', strategy_name + '.csv'))


def main():

    print("Loading processed files")
    # Load the signal file
    signal_df = pd.read_csv(join(INPUT_PATH, 'signal.csv'),
                            index_col=0)
    # Load stock monthly returns
    stock_returns = pd.read_csv(join(INPUT_PATH, 'monthly_returns.csv'),
                                index_col=0)
    # Load stock volatilties
    stock_vols = pd.read_csv(join(INPUT_PATH, 'monthly_vol.csv'),
                             index_col=0)
    # Load tickers with sectors
    sectors_sr = pd.read_csv(join(INPUT_PATH,  'sectors.csv'))

    sector_groupings = sectors_sr.reset_index().groupby(
        'Sector')['Symbol'].apply(lambda x: list(np.unique(x))).to_list()

    # Load covariances
    covariances_df = pd.read_csv(
        join(INPUT_PATH, 'monthly_covariances.csv'), index_col=[0, 1])

    """ Single stock strrategy """
    print("Running single stock strategy")
    strat = SingleStockStrategy(signal_df)
    strat.compute_weights()
    print("Strategy complete")
    dump_results(strat, stock_returns, 'single_stock')
    print("Dumped portfolio weights and returns to results folder")

    # Single Stock Strategy (Risk-Tolerance) : Single stock strategy with a restriction on risk tolerance
    print("Running single stock strategy with a annualized volatility constraint of 25%")
    strat = SingleStockStrategy(signal_df, stock_vols,
                                risk_constraint=True, vol_tolerance=0.25)
    strat.compute_weights()
    print("Strategy complete")
    dump_results(strat, stock_returns, 'single_stock_risktol')
    print("Dumped portfolio weights and returns to results folder")

    """Quintile Strategy: no constraint on single stock, but a long only constraint"""
    print("Running top quintile strategy with long-only constraint")
    strat = QuantileStrategy(signal_df, quantiles=5)
    strat.compute_weights()
    dump_results(strat, stock_returns, 'top_quantile')
    print("Strategy complete")
    print("Dumped portfolio weights and returns to results folder")

    """Quintile Strategy (Sector-Neutral): no constraint on single stock, but a long only and sector-neutrality constraints"""

    print("Running top quintile strategy with long-only and sector neutrality constraints")

    strat = QuantileStrategy(
        signal_df, quantiles=5, sector_neutral=True, sector_groupings=sector_groupings)
    strat.compute_weights()
    dump_results(strat, stock_returns, 'top_quantile_sector_neutral')
    print("Strategy complete")
    print("Dumped portfolio weights and returns to results folder")

    """Optimization strategy: no constraint of single stock, long only constraint"""

    response = input(
        "Do you want to run the optimization strategy as this will take a couple of minutes due to multiple optimizations? Please type Yes or No: ")

    if response == "Yes":
        print("Running optimization strategy with a target volatility of 15%")
        strat = OptimizationStrategy(signal_df, covariances_df, 0.15, True)
        strat.compute_weights()
        dump_results(strat, stock_returns, 'optimization')
        print("Strategy complete")
        print("Dumped portfolio weights and returns to results folder")
    else:
        print("Optimization strategy skipped")

    print("Program complete")


if __name__ == "__main__":

    main()
