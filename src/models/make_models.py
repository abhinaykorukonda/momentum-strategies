import pandas as pd
from strategies import SingleStockStrategy, QuantileStrategy
from os.path import join, dirname
import numpy as np


def main():

    processed_data_path = join(dirname(__file__), '..', '..')

    # Load the signal file
    signal_df = pd.read_csv(join(processed_data_path, 'data', 'processed', 'signal.csv'),
                            index_col=0)

    # Load stock monthly returns
    monthly_returns = pd.read_csv(join(processed_data_path, 'data', 'processed', 'monthly_returns.csv'),
                                  index_col=0)

    # Load stock volatilties
    stock_vols = pd.read_csv(join(processed_data_path, 'data', 'processed', 'monthly_vol.csv'),
                             index_col=0)

    # Load tickets with sectors

    sectors_sr = pd.read_csv(join(processed_data_path, 'data', 'processed', 'sectors.csv'))

    sector_groupings = sectors_sr.reset_index().groupby('Sector')['Symbol'].apply(lambda x: list(np.unique(x))).to_list()



    """ Strategy 1 : Single stock strrategy """

    # Strategy 1.1 : Single stock strategy with no restrictions on risk tolerance
    strat = SingleStockStrategy(signal_df)
    strat.compute_weights()
    strat_weights = strat.get_weights()
    strat_returns = strat.compute_returns(monthly_returns)

    # Strategy 1.2 : Single stock strategy with a restriction on risk tolerance
    # by using a limit of 25% annualized volaitlity on the stock
    strat = SingleStockStrategy(signal_df, stock_vols,
                                                   risk_constraint=True, vol_tolerance=0.25)
    strat.compute_weights()
    strat_weights = strat.get_weights()
    strat_returns = strat.compute_returns(monthly_returns)


    """Strategy 2: no constraint on single stock, but a long only constraint"""

    # Strategy 2.1 Trade top 5th quintile of the stocks based on the signal value
    strat = QuantileStrategy(signal_df,quantiles = 5)
    strat.compute_weights()
    strat_weights = strat.get_weights()
    strat_returns = strat.compute_returns(monthly_returns)


    """Strategy 2: no constraint on single stock, but a long only constraint"""

    # Strategy 2.1 Trade top 5th quintile of the stocks based on the signal value
    strat = QuantileStrategy(signal_df,quantiles = 5)
    strat.compute_weights()
    strat_weights = strat.get_weights()
    strat_returns = strat.compute_returns(monthly_returns)



if __name__ == "__main__":

    main()
