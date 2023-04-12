import numpy as np
import pandas as pd

def calc_sharpe(wts, ret):
	portfolio_ret = ret.mul(weights).sum(1)

	avg_portfolio_ret = portfolio_ret.mean()
	std_portfolio_ret = portfolio_ret.std()

	sharpe_portfolio = avg_portfolio_ret / std_portfolio_ret

	return sharpe_portfolio


def read_data(filepath):
	data  = pd.read_excel(filepath)
	data = data.set_index(pd.DatetimeIndex(data['Date']))

	etf_cols = ['SPY', 'IJS', 'GLD', 'TLT', 'SHY']
	mkt_factor_cols = ['VIX', 'Oil', 'Real GDP', 'Inflation', 'FFR']
	
	etf_data = data[etf_cols]
	mkt_factor_data = data[mkt_factor_cols]

	etf_ret = etf_data.pct_change(1)

	return etf_ret, mkt_factor_data


def wts_combinations(assets = 5, max_wt = 0.5, step_size = 0.05):
    total_sum = int(1/step_size)
    combinations = itertools.combinations_with_replacement(range(total_sum), assets)
    valid_combinations = []
    for c in combinations:
        valid = True
        for elem in c:
            if elem > total_sum / 2:
                valid = False
                break     
        if(valid and (sum(c) == total_sum)):
            valid_combinations.append(c)

    valid_permutations = []
    for c in valid_combinations:
        permutations = itertools.permutations(c)
        for p in permutations:
            if p not in valid_permutations:
                valid_permutations.append(p)

    all_combinations = []
    for p in valid_permutations:
        all_combinations.append(np.array(p) * step_size)
    
    return all_combinations

filepath = 'data.xlsx'
etf_ret, mkt_factor_data = read_data(filepath)

all_combinations = wts_combinations()

dataset = []
for date in etf_ret.index[:-700]:
    mkt_params = mkt_factor_data.loc[date].to_numpy()
    future_etf_ret = etf_ret.loc[date+pd.offsets.BDay(1) : date+pd.offsets.BDay(21)]
    
    for wts in all_combinations:
        params = np.concatenate((wts, mkt_params))
        future_sharpe = calc_sharpe(wts, future_etf_ret)
        dataset.append((params, future_sharpe))