Momentum Strategies
==============================

What is it?
--------------

A python project that builds and analyzes different momentum strategies based on financial stock data

Main Features
---------

Here are just few things this project can do

-  Prepares momentum signals based on T-12 to T-2 month returns of every stocks
-  Can build different portfolio strategies based on stock signal
    - Single Stock Strategy - Strategy to build portfolio weights based on hold only one stock every month rule
    - Quantile Strategy - Strategy to build portfolio weights by slicing stocks into quantiles based on the signal value
    - Optimization Strategy - Strategy that builds portfolio weights by optimizing the stock weights by maximizing the signal value with constraints on risk tolerance, long-only contraints 
- Portfolio strategies can be made sector neutral wherever possible
- Prepares portfolio analytic reports based on strategy returns and benchmark


Project Organization
------------

    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── results        <- The datasets that are outputs from the models.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   ├── report.md      <- The markdown report describing the strategy results
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── strategies.py  <- Class implementions for different strategies
    │   │
    │   ├── portfolio.py   <- Class implementions for tearsheet for a strategy
    │   │
    │   ├── make_data.py   <- Scripts to convert raw data to processed data for models
    │   │
    │   ├── make_models.py <- Scripts to train strategies and return results of the strategies
    │   │
    │   └── make_report.py <- Scripts to create exploratory and results oriented analytics and visualizations based on the results of the strategies
    ├── report.pdf         <- A full report discussing the assignment implementation and results
    

Usage
--------

Install the requirements

```
pip install -r requirements.txt
```

To convert the given raw dataset to processed datasets, run this command

```
py src/make_data.py
```

To train and prepare strategy model results, run this command

```
py src/make_models.py
```

To prepare visualization figures and strategy tearsheets, run this command

```
py src/make_report.py
```

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

