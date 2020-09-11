# Report

This report discusses the findings of different strategies using momentum signal. The daily stock prices of nearly 110 stocks for the past 10 years is given for our exercise. The report discusses the following strategies

- Single stock strategy
    - Without Volatlity constraint
    - With Volatility constraint

- Quantile stock strategy

- Optimization strategy

- Quantile stock strategy (sector-neutrality)


## Choice of momentum signal
----------------------------

While there are many ways to measure momentum, based on Jegadeesh 1990 and Fama-French website, the general momentum signal used is [2-12 momentum]((http://mba.tuck.dartmouth.edu/pages/faculty/ken.french/Data_Library/det_mom_factor.html)), (last 12-month returns, excluding the previous month) when examining intermediate-term momentum (last 12-month returns) effect on stock prices. The last month is excluded to avoid effects from [short-term return reversals](https://alphaarchitect.com/2015/01/14/quantitative-momentum-research-short-term-return-reversal/)

## Choice of benchmarks
-----------------------

For the strategies without sector neutrality, the best benchmark to be used is an equal weighted 