import pandas as pd
import numpy as np
import math

class BaseStrategy:

    def __init__(self, stock_signals):
        """Initializes the base strategy class

        Parameters
        ----------
        stock_signals : pandas.DataFrame
            Dataframe with columns as stock tickers, index as dates 
            and values as forecasted signals for the given time period
        """
        super().__init__()

        assert type(stock_signals) == pd.DataFrame 

        self._signal_df = stock_signals
        
        self._num_stocks = stock_signals.shape[1]
        self._num_months = stock_signals.shape[0]

        self._weights = None

    def set_stock_signals(self, stock_signals):
        """Resets the stock signals for the strategy

        Parameters
        -----------
        stock_signals : pandas.DataFrame
            Dataframe with columns as stock tickers, index as dates 
            and values as forecasted signals for the given time period
        """

        self._signal_df = stock_signals
        self._num_stocks = stock_signals.shape[1]
        self._num_months = stock_signals.shape[0]

    def get_stock_signals(self):
        """[summary]

        Returns
        -------
        pandas.DataFrame
            Dataframe with columns as stock tickers, index as month end dates 
            and values as forecasted signals for the given month
        """


        return self._signal_df

    def get_weights(self):
        """Returns the portfolio weights of all stocks based on the strategy and signal

        Returns
        -------
        pandas.DataFrame
            Dataframe with columns as stock tickers, index as month end dates 
            and values as portfolio weights for the stocks for the given month and stock
        """

        return self._weights

    def compute_returns(self,stock_returns):
        """Calculates the actual returns of the portfolio based on individual stock
         returns and portfolio weights trained by strategy and the signal 
        Returns
        -------
        pandas.Series
            Series with index as month end dates and values as monthly portfolio returns
        """

        assert type(stock_returns) == pd.DataFrame
        assert np.all(self._weights.index == stock_returns.index)
        assert np.all(self._weights.columns == stock_returns.columns)

        signal_returns = (self._weights * stock_returns).sum(axis = 1)

        return signal_returns


class SingleStockStrategy(BaseStrategy):


    def __init__(self, stock_signals, stock_vols = None, risk_constraint = False, vol_tolerance = 0.25):
        """Implements a single stock strategy that will trade only stock at any time


        Parameters
        ----------
        stock_signals : pandas.DataFrame
            Dataframe with columns as stock tickers, index as dates 
            and values as forecasted signals for the given time period
        stock_vols : pandas.DataFrame, optional
            Dataframe with columns as stock tickers, index as dates 
            and values as volatilites for the given time period, by default None
        risk_constraint : bool, optional
            Flag indicating whether a volatility constraint is present for the strategy, by default False
        vol_tolerance : float, optional
            Volatility tolerance for the portfolio if there is a risk contraint, by default 0.25
        """
    
        super().__init__(stock_signals)
        self._risk_constraint = risk_constraint
        self._vol_tolerance = vol_tolerance
        self._stock_vols = stock_vols
    
    def compute_weights(self):

        if self._risk_constraint:
            self._weights = self._compute_risk_tol_weights(self._vol_tolerance)

        else:
            self._weights = self._compute_no_risk_weights()

        return self._weights

    
    def _compute_no_risk_weights(self):

        return self._signal_df.T.apply(lambda x: x.index == x.idxmax()).T * 1

        
    def _compute_risk_tol_weights(self,vol_tolerance):

        assert type(vol_tolerance) in [int, float]
        assert type(self._stock_vols) == pd.DataFrame
        assert self._stock_vols.shape == self._signal_df.shape
        assert np.all(self._stock_vols.index == self._signal_df.index)
        assert np.all(self._stock_vols.columns == self._signal_df.columns)

        mask_df = self._stock_vols <= vol_tolerance
        self._weights = self._signal_df.T.apply(lambda x: x.index == self._custom_idxmax(x,mask_df.loc[x.name])).T

        return self._weights

    @staticmethod
    def _custom_idxmax(x, mask):

        x_copy = x.copy()
        x_copy.loc[~mask] = -math.inf

        if x_copy.max() == -math.inf:
            return x.idxmax()
        else:
            return x_copy.idxmax()


        

class QuantileStrategy(BaseStrategy):

    def __init__(self, stock_signals, quantiles, short_constraint = True, sector_neutral = False, sector_groupings = None):
        assert type(quantiles) == int 
        super().__init__(stock_signals)
        self._quantiles = quantiles
        self._quantile_cutoff = 1 - 1 / quantiles
        self._short_constraint = short_constraint
        self._sector_neutral = sector_neutral
        self._sector_groupings = sector_groupings

    
    def compute_weights(self):

        if self._short_constraint:
            self._weights = self._compute_long_only_weights()
        else:
            self._weights = self._compute_long_short_weights()

        return self._weights


    def _compute_long_short_weights(self):
        # TODO 
        raise NotImplementedError

    def _compute_long_only_weights(self):

        positions = self._signal_df.apply(lambda x: x >= np.quantile(x.dropna(), 0.8),
            axis=1) * 1
        weights = positions.divide(positions.sum(axis = 1), axis = 0)

    
    def _compute_long_only_weights_sector_neutral(self):
        
        sector_df_list = [self._signal_df.loc[:,self._signal_df.columns.isin(group)] for group in self._sector_groupings]

        sector_strats = [QuantileStrategy(df,self._quantiles) for df in sector_df_list]

        sector_weights = []
        for strat in sector_strats:
            strat.compute_weights()
            sector_weights.append(strat.get_weights())
        return sector_weights




    
        






    