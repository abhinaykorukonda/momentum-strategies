# Assignment Report


1. [Choice of momentum signal](#choice-of-momentum-signal)
2. [Choice of benchmarks](#choice-of-benchmarks)
    1. [No Sector Neutrality](#no-sector-neutrality)
    2. [Sector Neutrality](#sector-neutrality)
3. [Backtest Assumptions](#backtest-assumptions)
4. [Strategy Choices](#strategy-choices)
5. [Technical Implementation](#technical-implementation)


This report discusses the findings of different strategies using momentum signal. The daily stock prices of nearly 110 stocks for the past 10 years is given for our exercise. The report discusses the following strategies

- Single stock strategy
    - Without Volatlity constraint
    - With Volatility constraint

- Quantile stock strategy

- Optimization strategy

- Quantile stock strategy (sector-neutrality)


## Choice of momentum signal


While there are many ways to measure momentum, based on Jegadeesh 1990 and Fama-French website, the general momentum signal used is [2-12 momentum](http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html), (last 12-month returns, excluding the previous month) when examining intermediate-term momentum (last 12-month returns) effect on stock prices. The last month is excluded to avoid effects from [short-term return reversals](https://alphaarchitect.com/2015/01/14/quantitative-momentum-research-short-term-return-reversal/)

## Choice of benchmarks


### No sector neutrality
For the strategies without sector neutrality, the best benchmark to be used is an equal weighted index. Many institutions use market-cap weighted benchmarks such as S\&P 500 to measure their portfolio.

Why not use market-cap weighted benchmark in that case? Market-cap weighted benchmarks experience effects due to the SMB factor also known as the [size factor](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Market-cap weighted benchmarks give more weight to bigger market-cap stocks. Historically, the momentum factor experienced [negative correlations](https://www.factorresearch.com/research-factor-correlation-check) of nearly -13\% to the size factor. So when we compare our momentum strategy to a market-cap weighted benchmark, noises due to size factor might exist. Equal weighted benchmark will avoid this problem. The equal weighted benchmark definitely fails when there are stocks with very low volume influencing the entire universe of stocks. Since our universe only contains 110 stocks, using an equal weighted benchmark will not be a problem.


### Sector neutrality
In case of sector neutrality, the best benchmark to be used is a variation of equal weighted benchmark. Since each sector will have equal weight, we should ensure that the benchmark also has equal weights in the sectors. To achieve this, we will assign a weight of 1/11 (there are total of 11 sectors) to each sector, and among each sector we will weigh the stocks equally

This ensures that we have an apples to apples comparison when strategies with sector neutrality are considered

## Backtest Assumptions
The following are the backtest assumptions used for some or all of the strategies

- Every strategy will be rebalanced monthly to avoid excessive trading
- Risk free rate is assumed to be 0
- All portfolios will be fully-invested with no investment in risk-free asset
- No transaction costs are assumed
- Covariances, variances, and volatilities are calculated using past 3-year daily returns history
- Geometric returns are used to compute stock returns instead of log returns (Even though, the log returns are popular in academics, the practical applications are not applicable)
- Risk tolerance of a strategy is defined by target annualized volatlity
- Backtest starts from 2013-10-31
- Results are posted considering all the months after 2013-10-31


The following assumptions/considerations are NOT followed
- There is no testing set defined, even though the right way to test our results is to choose the best strategy and test it on unseen test data set at the end, we chose to use the entire dataset for exploring our results. The right way to do it is definitely to have a unseen test set to report results based on the best strategy in cross-validation set.  


## Strategy choices


### Single stock strategy

Based on the problem statement, the two contraints are 

1) You can only be invested in one asset at any point in time or none
2) There is no shorting allowed

Based on these constraints, there are two strategies that can be built

- Every month, pick one single stock with the highest momentum signal and trade it for that month
- Every month, pick one single stock with the highest momentum signal with a constraint on the volatility of less than or equal to some threshold value and trade it for that month


Usually, momentum is highly correlated to volatility as higher-volatlity stocks tend to have higher returns (risk-reward tradeoff). As we are only picking one stock, there is very little diversification benefit, the best way to control idiosyncratic and beta risks is to select a stock only if its volatility is below a certain threshold. The threshold here is considered to be 25\% as it's the median annualized volatility of all stocks

### Quantile strategy

When the first constraint is removed, the only constraint is on the shorting

Most factor strategies are dollar-neutral. The universe of stocks are sorted based on the signal value and broken into n-quantiles. An equal weighted portfolio of the top quantile is bought and another equal weighted portfolio of bottom quantile is shorted. The long-short strategy ensures that the market beta effect is reduced as much as possible. 

Since our constraint cannot allow us to do that, we can simply trade the equal weighted portfolio of stocks in the top quantile every month

For this exercise, we chose to break the universe into 5 quantiles every month based on the momentum signal value


### Optimization strategy

Another way to target this problem when the first constraint is removed is to use portfolio optimization

Instead of choosing the equal weighted top quantile, one can find a dynamic portfolio weights that will maximize the value of the ex-ante alpha with constraints on ex-ante portfolio volatility and long-only trading

Based on pg 383 of Kahn 2000, the following equation shows us how to set up the problem 



This is a simple quadratic optimization problem with linear constraints. Every month, this optimization will be done based on the new signal values (or ex-ante alphas) and the updated covariance matrix. 

Even though an appropriate benchmark would be market weighted benchmark (see [efficient-frontier](https://en.wikipedia.org/wiki/Efficient_frontier)), we will still use equal-weighted benchmark for our analysis


### Quantile Strategy (Sector-Neutral)

With the first constraint and a new constraint on sector-neutrality (equal weights) added, we can simply modify the Quantile Strategy to conform with the new constraint

Instead of sorting the stocks globally based on the signal, we can sort the stocks locally in the sector and pick the portfolio of top nth quantile in each sector. So, the quantile strategy can be implemented for every sector individually and then combined into global portfolio strategy.

By doing a quantile strategy in every sector, we will end up with 11 long-only portfolios. We will then make sure to equally weigh each sector by multiplying the portfolio weights with 1/11 (1 / no. of sectors). This ensures that the weights of the sectors are equal and the total sum of the portfolio weights is equal to 1.

## Technical Implementation

The project is executed using 3 important scripts and two modules

Scripts:
1. `src/make_dataset.py`
2. `src/make_models.py`
3. `src/make_report.py`

Modules:
1. `src/strategies.py`
2. `src/portfolio.py`


In order to conduct our analysis, we need to convert our raw data(`data/raw/qr_lead_hw_data_candidate.csv.gz`) into datasets that are suitable for modeling using the `src/make_dataset.py` script


To run this script, run the following command

```
python3 src/make_dataset.py
```

The python script [make_dataset.py](../src/data/make_dataset.py) converts the raw data into the following important csv files



- `data/processed/monthly_returns.csv` - This file stores monthly returns of each stock with the columns as stock tickers, rows as monthly dates and values as the monthly returns corresponding to the row and column. The monthly returns are calculated using adjusted close prices for accurate represntation of returns adjusted for dividends, stock splits etc.

- `data/processed/signal.csv` - This file stores the signal values of each stock with the columns as stock tickers, rows as monthly dates and values as the monthly returns corresponding to the row and column

- `data/processed/monthly_vol.csv` - This file stores monthly annualized volatilities using past 3 years history of each stock with the columns as stock tickers, rows as monthly dates and values as the annualized vaolatility corresponding to the row and column

- `data/processed/sectors.csv` - This file contains two columns, the first column represents the stock ticker and the second column reprsents the sector it belongs to

- `data/processed/eqwt_benchmark_returns.csv` - This is the equal weighted benchmark monthly time series of returns

- `data/processed/sector_eqwt_benchmark_returns.csv` - This is the sector adjusted equal weighted benchmark monthly time series of returns


After these files are created, the next script that needs to be run is `src/make_models.py`. This script builds the strategies, computes the portfolio weights for every month and calculates the realized portfolio returns using the stock returns provided

To run the full script, run the following commmand

```
python3 src/make_models.py
```

This script uses the `src/strategies.py` module that has class implementations of different strategies

As an example, a single stock strategy can be implemented by using the following block of code

```
import pandas as pd
from strategies import SingleStockStrategy

signal_df = pd.read_csv('data/processed/signal.csv',index_col = 0,parse_dates = True)
stock_returns = pd.read_csv('data/processed/stock_returns.csv', index_col = 0, parse_dates = True)

# Initialize a strategy and calculate portfolio weights and realized returns every month
strat = SingleStockStrategy(signal_df) 
strat.compute_weights()
portfolio_weights = strat.get_weights()
portfolio_returns = strat.compute_returns(stock_returns)
```



|                       |   single_stock |   single_stock_risktol |   benchmark |
|:----------------------|---------------:|-----------------------:|------------:|
| Mean                  |     0.578697   |             0.228993   |   0.144446  |
| Volatility            |     0.465486   |             0.243748   |   0.14924   |
| Sharpe                |     1.24321    |             0.939466   |   0.967876  |
| Skew                  |    -0.208676   |             0.538185   |  -0.708757  |
| Kurtosis              |     0.894886   |             0.904876   |   3.94464   |
| Adjusted Sharpe       |     1.11781    |             0.98737    |   0.708194  |
| Drawdown (%)          |    -0.447743   |            -0.302778   |  -0.254575  |
| Beta                  |     0.109278   |             0.300963   |   1         |
| Alpha                 |     0.0812071  |             0.0755279  |   0         |
| IR                    |     0.176467   |             0.335539   | nan         |
| Turnover (%)          |     6.07229    |            11.5663     | nan         |
| p-values(Adj. Sharpe) |     0.00212658 |             0.00556498 |   0.0330341 |
| p-values(IR)          |     0.321895   |             0.19004    | nan         |


|                               |   Single Stock (risk-tol) |   Top Quantile |   Optimization |   Top Quantile (sector-neutral) |   Eq. wt. Benchmark |
|:------------------------------|--------------------------:|---------------:|---------------:|--------------------------------:|--------------------:|
| Single Stock (risk-tol)       |                      1    |           0.66 |           0.68 |                            0.56 |                0.49 |
| Top Quantile                  |                      0.66 |           1    |           0.86 |                            0.9  |                0.84 |
| Optimization                  |                      0.68 |           0.86 |           1    |                            0.79 |                0.67 |
| Top Quantile (sector-neutral) |                      0.56 |           0.9  |           0.79 |                            1    |                0.93 |
| Eq. wt. Benchmark             |                      0.49 |           0.84 |           0.67 |                            0.93 |                1    |