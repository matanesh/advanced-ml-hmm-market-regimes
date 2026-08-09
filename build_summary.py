"""
Build summary of all experiments with directory names and filter for HMM wins.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import json
import pandas as pd

def build_summary(base_dir='experiments_extended'):
    summary_rows = []
    for dirname in sorted(os.listdir(base_dir)):
        if dirname.startswith('run_'):
            dirpath = os.path.join(base_dir, dirname)
            config_path = os.path.join(dirpath, 'config.json')
            results_path = os.path.join(dirpath, 'results.csv')
            if os.path.exists(config_path) and os.path.exists(results_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                ticker = config['ticker']
                df = pd.read_csv(results_path)
                # Results are sorted by DPA descending (as we saved them)
                best_row = df.iloc[0]
                # Also get naive train mean return DPA for comparison
                naive_row = df[df['model'] == 'Naive - train mean return']
                naive_dpa = naive_row.iloc[0]['DPA_direction_accuracy'] if not naive_row.empty else None
                summary_rows.append({
                    'dirname': dirname,
                    'ticker': ticker,
                    'best_model': best_row['model'],
                    'best_dpa': best_row['DPA_direction_accuracy'],
                    'best_mae': best_row['MAE_return'],
                    'best_rmse': best_row['RMSE_return'],
                    'best_mape': best_row['MAPE_price_%'],
                    'best_ll_train': best_row['log_likelihood_train'],
                    'best_ll_test': best_row['log_likelihood_test'],
                    'naive_dpa': naive_dpa,
                    'dpa_improvement_over_naive': best_row['DPA_direction_accuracy'] - naive_dpa if naive_dpa is not None else None
                })
            else:
                print(f'Missing config or results in {dirname}')
    summary_df = pd.DataFrame(summary_rows)
    # For each ticker, keep only the row with the highest best_dpa (in case of multiple runs)
    summary_df = summary_df.sort_values('best_dpa', ascending=False).drop_duplicates('ticker', keep='first')
    return summary_df

if __name__ == '__main__':
    summary_df = build_summary()
    print(f'Total unique tickers: {len(summary_df)}')
    print('\\nBest models distribution:')
    print(summary_df['best_model'].value_counts())
    print('\\nSaving summary to experiments_extended/summary_with_dirname.csv')
    summary_df.to_csv('experiments_extended/summary_with_dirname.csv', index=False)
    print('\\nHMM winning cases:')
    hmm_wins = summary_df[summary_df['best_model'].str.startswith('Gaussian HMM')]
    print(hmm_wins[['ticker', 'best_model', 'best_dpa', 'dpa_improvement_over_naive']])
    # Save list of winning cases for further analysis
    hmm_wins.to_csv('experiments_extended/hmm_winning_cases.csv', index=False)
    print('\\nSaved winning cases to experiments_extended/hmm_winning_cases.csv')