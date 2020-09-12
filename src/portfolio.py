import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis, linregress, t
import math


class TearSheet:
    """A low-level class that will generate a tearsheet given strategy and
    benchmark returns, portfolio weights
    """

    def __init__(self, strategy_returns, weights, benchmark_returns, frequency='monthly'):
        """Initializes the tearsheet

        Parameters
        ----------
        strategy_returns : dict{str: pd.Series}
            A dictionary of strategy return series with each series has an index representing the dates,
            values as the portfolio returns between the dates 
        weights : dict{str: pd.DataFrame}
            A dictionary of dataframe with each dataframe representing the dates, columns as stock tickers and 
            values as the portfolio weights of the strategy between the dates 
        benchmark_returns : pd.Series
            A series returns with same index as each series in the strategy returns
        frequency : str, ['daily','monthly','yearly']
            Frequency of the returns in the strategy, by default 'monthly'
        """

        self._strat_rets = strategy_returns
        self._bench_rets = benchmark_returns
        self._weights = weights

        self._results = None
        if frequency == 'monthly':
            self._time_factor = 12
        elif frequency == 'daily':
            self._time_factor = 252
        else:
            self._time_factor = 1

    def get_results(self):
        """Gets the portfolio results with all the tearsheet metrics

        Returns
        -------
        pd.DataFrame
            Returns a dataframe of tearsheet results of the strategies along with the benchmark
        """

        if self._results is not None:
            return self._results

    def compute_results(self):
        """Computes the tearsheet results for all strategies with benchmark
        """

        benchmark = self._bench_rets
        time_factor = self._time_factor

        funcs = [('Mean', lambda x: self.mean(x, time_factor)),
                 ('Volatility', lambda x: self.vol(x, time_factor)),
                 ('Sharpe', lambda x: self.sharpe_ratio(x, time_factor)),
                 ('Skew', skew),
                 ('Kurtosis', kurtosis),
                 ('Adjusted Sharpe', lambda x: self.adj_sharpe_ratio(x, time_factor)),
                 ('Drawdown (%)', self.draw_down),
                 ('Beta', lambda x: self.beta(x, benchmark)),
                 ('Alpha', lambda x: self.alpha(x, benchmark, time_factor)),
                 ('IR', lambda x: self.IR(x, benchmark, time_factor)),
                 ]

        def agg_func(x): return [func(x) for func_name, func in funcs]

        returns_df = pd.DataFrame(dict(self._strat_rets))

        returns_df['benchmark'] = benchmark
        results = returns_df.apply(agg_func, result_type='expand')

        results.index = [func_name for func_name, _ in funcs]

        turnover = pd.Series({key: self.turnover(weights, self._time_factor)
                              for key, weights in self._weights.items()})

        self._results = pd.concat([results, pd.DataFrame(
            turnover).T.rename({0: 'Turnover (%)'})], axis=0)

        num_months = benchmark.shape[0]

        mean_p_values = self.p_value(
            self._results.loc['Adjusted Sharpe', :], num_months, self._time_factor)

        self._results.loc['p-values(Adj. Sharpe)', :] = mean_p_values

        ir_p_values = self.p_value(
            self._results.loc['IR', :], num_months, self._time_factor)

        self._results.loc['p-values(IR)', :] = ir_p_values

    @staticmethod
    def sharpe_ratio(x, time_factor):
        """Calculates sharpe ratio

        Parameters
        ----------
        x : pd.Series
        time_factor : int
            A time factor to get annualized Sharpe ratio 

        Returns
        -------
        float

        """
        return x.mean() * np.sqrt(time_factor) / x.std()

    @staticmethod
    def adj_sharpe_ratio(x, time_factor):
        """Calculates sharpe ratio adjusted for skewness and kurtosis

        Parameters
        ----------
        x : pd.Series
        time_factor : int
            A time factor to get annualized Sharpe ratio 

        Returns
        -------
        float
        """
        sharpe = x.mean() * np.sqrt(time_factor) / x.std()
        skewness = skew(x)
        excess_kurtosis = kurtosis(x)
        adj_sharpe = sharpe * (1 + skewness*sharpe/6 -
                               excess_kurtosis * (sharpe**2)/24)
        return adj_sharpe

    @staticmethod
    def draw_down(x):
        """Calcualtes maximum drawdown of a return series

        Parameters
        ----------
        x : pd.Series
            Return series of a strategy or a benchmark

        Returns
        -------
        float
            maximum drawdown of a strategy or a benchmark
        """

        x_copy = x.copy()
        x_cum_rets = (x_copy + 1).cumprod()

        running_max = np.maximum.accumulate(x_cum_rets)

        return ((x_cum_rets / running_max) - 1).min()

    @staticmethod
    def mean(x, time_factor):
        """Calculates annualized mean

        Parameters
        ----------
        x : pd.Series
        time_factor : int
            A time factor to scale the mean according to the frequency

        Returns
        -------
        float
        """
        return x.mean() * time_factor

    @staticmethod
    def vol(x, time_factor):
        """Calculates annualized volatility

        Parameters
        ----------
        x : pd.Series
        time_factor : int
            A time factor to scale the volatility according to the frequency

        Returns
        -------
        float
        """

        return x.std() * np.sqrt(time_factor)

    @staticmethod
    def alpha(x, benchmark, time_factor):
        """Calculates the alpha of a strategy comparing to benchmark using 
        linear regrression

        Parameters
        ----------
        x : pd.Series
            Strategy return series
        benchmark : pd.Series
            Benchmark return series for which the strategy is compared
        time_factor : int
            A time factor to scale the volatility according to the frequency

        Returns
        -------
        float
        """
        results = linregress(x, benchmark)
        return results.intercept * time_factor

    @staticmethod
    def IR(x, benchmark, time_factor):
        """Calculates the information ratio of a strategy comparing to benchmark using 
        linear regrression

        Parameters
        ----------
        x : pd.Series
            Strategy return series
        benchmark : pd.Series
            Benchmark return series for which the strategy is compared
        time_factor : int
            A time factor to scale the volatility according to the frequency

        Returns
        -------
        float
        """
        results = linregress(x, benchmark)
        slope = results.slope
        intercept = results.intercept
        residuals = x - slope * benchmark - intercept
        resid_vol = residuals.std()

        if resid_vol == 0:
            return np.NaN

        return intercept * np.sqrt(time_factor) / resid_vol

    @staticmethod
    def turnover(weights, time_factor):
        """Calculates the annualized turnover of a strategy given the time factor

        Parameters
        ----------
        weights : pd.DataFrame
            A dataframe representing the dates, columns as stock tickers and 
            values as the portfolio weights of the strategy between the dates 
        time_factor : int
            A time factor to scale the volatility according to the frequency

        Returns
        -------
        float
        """

        return np.abs(weights.diff()).sum(axis=1).mean() * time_factor

    @staticmethod
    def beta(x, benchmark):
        """Calculates the beta of a strategy comparing to benchmark using 
        linear regrression

        Parameters
        ----------
        x : pd.Series
            Strategy return series
        benchmark : pd.Series
            Benchmark return series for which the strategy is compared
        time_factor : int
            A time factor to scale the volatility according to the frequency

        Returns
        -------
        float
        """

        results = linregress(x, benchmark)
        return results.slope

    @staticmethod
    def p_value(sharpe, count, time_factor):
        """Calculates the p-value of a strategy based on the annualized sharpe/information ratio,
        number of observations and the time-factor to scale back the ratio

        Parameters
        ----------
        sharpe : float
            Annualized sharpe or information ratio
        count : int
            Number of observations used for estimating the sharpe ratio
        time_factor : int
            A time factor to scale the volatility according to the frequency

        Returns
        -------
        float
        """

        sharpe_copy = sharpe.copy()
        sharpe_copy = sharpe_copy.fillna(0)

        return (1 - t.cdf(sharpe_copy * np.sqrt(count) / np.sqrt(time_factor), count))
