import pandas as pd
import numpy as np
from pathlib import Path

from loading import load_all_stocks
from alpha_calculator import calculate_all_alphas
from features import prepare_features_per_asset, split_data_per_asset, train_all_models
from backtest import run_backtest, calculate_metrics, plot_results

# ========== КОНФИГУРАЦИЯ ==========
DATA_PATH = '/Users/ivangolovachev/vs projects/alpha_portfolio/data'
FIXED_WEIGHTS = [0.05, 0.1, 0.15, 0.2, 0.5]  # для 5 акций
COMMISSION = 0.0001

# Настройки модели
MODEL_TYPE = 'lasso'  # 'linear', 'ridge', 'lasso'
USE_PCA = True
N_COMPONENTS = 5
ALPHA = 5.0  # для ridge/lasso

# ========== 1. ПОДГОТОВКА ДАННЫХ ==========

stocks = load_all_stocks(DATA_PATH)
stocks_with_alphas = calculate_all_alphas(stocks)
features = prepare_features_per_asset(stocks_with_alphas)
train_dict, val_dict, test_dict = split_data_per_asset(features)

# ========== 2. ОБУЧЕНИЕ МОДЕЛЕЙ ==========

models, feature_cols_dict = train_all_models(
    train_dict, 
    model_type=MODEL_TYPE,
    use_pca=USE_PCA,
    n_components=N_COMPONENTS,
    alpha=ALPHA
)


# ========== 3. БЭКТЕСТ ==========

dates, strategy_returns, benchmark_returns = run_backtest(
    models, val_dict, feature_cols_dict, FIXED_WEIGHTS, COMMISSION
)

# ========== 4. РЕЗУЛЬТАТЫ ==========

strategy_cumprod = calculate_metrics(strategy_returns, "Ребалансировка")
benchmark_cumprod = calculate_metrics(benchmark_returns, "Равные веса")

# ========== 5. ГРАФИК ==========
plot_results(dates, strategy_cumprod, benchmark_cumprod)



# results_df = pd.DataFrame({
#     'date': dates,
#     'strategy': strategy_returns,
#     'benchmark': benchmark_returns
# })
# results_df.to_csv('backtest_results.csv', index=False)