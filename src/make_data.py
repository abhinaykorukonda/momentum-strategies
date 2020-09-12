# -*- coding: utf-8 -*-
from pathlib import Path
import pandas as pd
import numpy as np
from os.path import join, dirname


CUTOFF_DATE = '2013-10-31'


def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """

    input_file_path = join(dirname(__file__), '..',
                           'data', 'raw', 'qr_lead_hw_data_candidate.csv.gz')

    # Load the raw dataset
    # Use "C" engine to load the data faster
    print("loading raw file")
    raw_df = pd.read_csv(input_file_path,
                         index_col=0, parse_dates=['Date'], engine='c')
    print("raw file read sucessfully")

    print("Preparing to process the files in processed folder")

    # Pivot the data where columns are stocks, rows are Dates and values are adjusted close prices
    pivoted_df = raw_df.pivot(
        index='Date', columns='Symbol', values='Adj Close')

    # Get geometric daily returns from the stocks
    daily_returns = np.exp(np.log(pivoted_df).diff(1)) - 1

    # Get geometric monthly returns from the stocks
    monthend_returns = np.exp(
        np.log(pivoted_df).resample('M').last().diff(1)) - 1

    # Get annualized volatilties everyday using past 3 years data
    daily_vol = daily_returns.rolling('756d').std() * np.sqrt(252)

    # Collect only the monthly volatilities
    monthly_vol = daily_vol.resample('M').last()

    # Get monthly covariance matrices, we do not need daily covariance matrices,
    # so we will write a simple for loop to get the monthly covariance matrices
    covariances = {}
    # Get month end dates for the daily returns
    monthend_dates = daily_returns.resample('M').last().index

    # Collect the covariance matrices from the year 3 using past 3 years
    for month_end in monthend_dates[36:]:

        mask = (daily_returns.index <= month_end) & (
            daily_returns.index >= (month_end - pd.DateOffset(years=3)))

        cov_matrix = (daily_returns.loc[mask, :].cov()*252).to_dict('list')

        covariances[month_end] = cov_matrix

    # The following lines of code converts the covariance matrices to a multi index dataframe
    # The level 1 index is the data, level 2 index is the stock ticker
    # The columns are stock tickers as well, where each value represents covariance of both stocks
    cov_df = pd.DataFrame.from_dict(
        covariances, orient="index").stack().to_frame()
    cov_df = pd.DataFrame(cov_df[0].values.tolist(), index=cov_df.index)
    cov_df.columns = daily_returns.columns

    # Dump daily returns
    # print("Dumping daily returns file as daily_returns.csv")
    # daily_returns.loc[daily_returns.index >= CUTOFF_DATE,
    #                   :].to_csv('data/processed/daily_returns.csv')

    print("Dumping covariance matrices as monthly_covariances.csv")
    cov_df.to_csv('data/processed/monthly_covariances.csv')

    print("Dumping month end returns as monthly_returns.csv")
    monthend_returns.loc[monthend_returns.index >= CUTOFF_DATE, :].to_csv(
        'data/processed/monthly_returns.csv')

    print("Dumping monthly volatilities as monthly_vol.csv")
    monthly_vol.loc[monthly_vol.index >= CUTOFF_DATE,
                    :].to_csv('data/processed/monthly_vol.csv')

    # Make momentum signal based T-12 to T-2 month while leaving out last month

    signal = monthend_returns.shift(2).rolling(
        11).mean().dropna(axis=0, how='all') * 12

    print("Dumping stock signals as signal.csv")
    signal.loc[signal.index >= CUTOFF_DATE, :].to_csv(
        'data/processed/signal.csv')

    # Make benchmarks
    # First benchmark (market cap weighted return)

    # Get monthly volumes of all stocks
    volumes_df = raw_df.pivot(
        index='Date', columns='Symbol', values='Volume').resample('M').last()

    # Compute benchmark returns
    mcap_benchmark_returns = (
        volumes_df * monthend_returns).sum(axis=1).divide(volumes_df.sum(axis=1))

    # Compute benchmark valatilities
    # mcap_benchmark_vol = (mcap_benchmark_returns.rolling(
    #     36).std() * np.sqrt(12)).dropna()

    # print("Dumping market-cap benchmark monthly returns and volatilities \
    #     as mcap_benchmark_returns.csv and mcap_benchmark_vol.csv")
    # mcap_benchmark_returns.loc[mcap_benchmark_returns.index >= CUTOFF_DATE].to_csv(
    #     'data/processed/mcap_benchmark_returns.csv')
    # mcap_benchmark_vol.loc[mcap_benchmark_vol.index >= CUTOFF_DATE].to_csv(
    #     'data/processed/mcap_benchmark_vol.csv')

    # Equal weighted benchmark
    eqwt_benchmark_returns = monthend_returns.mean(axis=1)
    eqwt_benchmark_vol = (monthend_returns.rolling(
        36).std() * np.sqrt(12)) * np.sqrt(12)

    print("Dumping equal weighted benchmark monthly returns and volatilities \
         as eqwt_benchmark_returns.csv and eqwt_benchmark_vol.csv")
    eqwt_benchmark_returns.loc[eqwt_benchmark_returns.index >= CUTOFF_DATE].to_csv(
        'data/processed/eqwt_benchmark_returns.csv')
    eqwt_benchmark_vol.loc[eqwt_benchmark_vol.index >= CUTOFF_DATE].to_csv(
        'data/processed/eqwt_benchmark_vol.csv')

    # Equal sector weighted benchmark
    # Get all sector groupings
    sectors = raw_df[['Symbol', 'Sector']
                     ].drop_duplicates().set_index('Symbol')

    print("Dumping sector groupings as sectors.csv")
    sectors.to_csv('data/processed/sectors.csv')

    sector_counts = pd.DataFrame(raw_df.groupby(
        ['Date', 'Sector'])['Symbol'].count())

    sector_counts.rename(columns={'Symbol': 'Count'}, inplace=True)

    # Third benchmark
    stock_weights = (1 / (raw_df.merge(sector_counts,
                                       on=['Date', 'Sector'],
                                       how='left').pivot('Date', 'Symbol', 'Count')) / 11).resample('M').last()

    sector_eq_benchmark_returns = (
        stock_weights * monthend_returns).sum(axis=1)
    sector_eq_benchmark_vol = (sector_eq_benchmark_returns.rolling(
        36).std() * np.sqrt(12)) * np.sqrt(12)

    print("Dumping sector equal weighted benchmark monthly returns and volatilities \
         as sector_eqwt_benchmark_returns.csv and sector_eqwt_benchmark_vol.csv")
    sector_eq_benchmark_returns.loc[sector_eq_benchmark_returns.index >= CUTOFF_DATE].to_csv(
        'data/processed/sector_eqwt_benchmark_returns.csv')
    sector_eq_benchmark_vol.loc[sector_eq_benchmark_vol.index >= CUTOFF_DATE].to_csv(
        'data/processed/sector_eqwt_benchmark_vol.csv')


if __name__ == '__main__':

    main()
