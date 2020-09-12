import pandas as pd
import numpy as np
from os.path import join, dirname
from collections import OrderedDict
from portfolio import TearSheet
import matplotlib.pyplot as plt

plt.style.use('seaborn-whitegrid')


INPUT_PATH = join(dirname(__file__), '..', 'data', 'results')

OUTPUT_PATH = join(dirname(__file__), '..', 'reports')


def main():

    strategies = {'single_stock': 'Single Stock',
                  'single_stock_risktol': 'Single Stock (risk-tol)',
                  'top_quantile': 'Top Quantile',
                  'optimization': 'Optimization',
                  'top_quantile_sector_neutral': 'Top Quantile (sector-neutral)'}

    weights_df_lst = []
    returns_df_lst = []
    # Load the portfolio weights and returns
    for key, _ in strategies.items():

        monthly_returns = pd.read_csv(
            join(INPUT_PATH, 'portfolio_returns', key + '.csv'),
            index_col=0, squeeze=True, parse_dates=True)

        returns_df_lst.append((key, monthly_returns))

        weights = pd.read_csv(join(
            INPUT_PATH, 'portfolio_weights', key + '.csv'),
            index_col=0, parse_dates=True)
        weights_df_lst.append((key, weights))

    # Load the benchmarks
    benchmark_returns = pd.read_csv(join(
        INPUT_PATH, '..', 'processed', 'eqwt_benchmark_returns.csv'), index_col=0, parse_dates=True, squeeze=True)
    sector_benchmark_returns = pd.read_csv(
        join(INPUT_PATH, '..', 'processed', 'sector_eqwt_benchmark_returns.csv'), index_col=0, parse_dates=True, squeeze=True)

    # Prepare tearsheet for first group
    tearsheet_one = TearSheet(dict(returns_df_lst[0:4]),
                              dict(weights_df_lst[0:4]),
                              benchmark_returns, frequency='monthly')

    tearsheet_one.compute_results()

    # Prepare tearsheet for second group
    tearsheet_two = TearSheet(dict(returns_df_lst[4:]),
                              dict(weights_df_lst[4:]),
                              sector_benchmark_returns, frequency='monthly')

    tearsheet_two.compute_results()

    # Get tearsheet results
    tearsheet_one_results = tearsheet_one.get_results()
    tearsheet_two_results = tearsheet_two.get_results()

    # Rename columns in tearsheets
    tearsheet_one_results = tearsheet_one_results.rename(columns=strategies).rename(
        columns={'benchmark': 'Eq. wt benchmark'})

    tearsheet_two_results = tearsheet_two_results.rename(columns=strategies).rename(
        columns={'benchmark': 'Sector eq. wt benchmark'})

    # Combine both the tearsheets
    combined_tearsheet = pd.concat(
        [tearsheet_one_results, tearsheet_two_results], axis=1)

    # Dump the tearsheet as a csv file
    print("Dump combined tearsheet as reports/tearsheet.csv")
    combined_tearsheet.round(4).to_csv(join(OUTPUT_PATH, 'tearsheet.csv'))
    print("Here is the tearsheet")
    print("----------------------------------")
    print(combined_tearsheet.round(4).fillna(pd.NA).to_markdown())

    """Collect cumulative returns"""
    monthly_returns = pd.DataFrame(dict(returns_df_lst))
    monthly_returns['Eq. wt. Benchmark'] = benchmark_returns
    monthly_returns.rename(columns=strategies, inplace=True)
    cumulative_returns = (1 + monthly_returns).cumprod()

    # Plot cumulative returns for select strategies

    subset_plot = ['Single Stock (risk-tol)', 'Top Quantile',
                   'Optimization', 'Top Quantile (sector-neutral)', 'Eq. wt. Benchmark']

    (cumulative_returns[subset_plot] * 100).plot()
    plt.title('Cumulative returns of strategies')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.tight_layout()
    plt.gcf().set_size_inches(15, 8)
    plt.savefig(join(OUTPUT_PATH, 'figures',
                     'cumulative_returns.png'), dpi=400)

    print("Saving cumulative returns plot at reports/figures/cumulative_returns.png")

    # Plot 3 year rolling sharpe ratio
    rolling_sharpe = monthly_returns.rolling(36, min_periods=36).mean(
    ) * np.sqrt(12) / monthly_returns.rolling(36, min_periods=36).std()

    (rolling_sharpe[subset_plot].dropna()).plot()
    plt.title('3 year rolling sharpe ratio')
    plt.xlabel('Date')
    plt.ylabel('Sharpe ratio')
    plt.tight_layout()
    plt.gcf().set_size_inches(15, 8)
    plt.savefig(join(OUTPUT_PATH, 'figures', 'rolling_sharpe.png'), dpi=400)
    print("Saving 3 year rolling sharpe plot at reports/figures/rolling_sharpe.png")

    # Plot density plot of monthly returns
    monthly_returns[subset_plot].plot.kde()
    plt.title('Density plot of monthly returns')
    plt.xlabel('Monthly returns')
    plt.tight_layout()
    plt.gcf().set_size_inches(10, 8)
    plt.savefig(join(OUTPUT_PATH, 'figures', 'density.png'), dpi=400)
    print("Saving density plot at reports/figures/rolling_sharpe.png")


    # Plotting correlations

    print(monthly_returns[subset_plot].corr().round(2).to_markdown())



if __name__ == "__main__":
    main()
