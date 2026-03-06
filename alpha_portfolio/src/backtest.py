import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_weights_by_rank(predictions, fixed_weights):
    sorted_tickers = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    weights = {}
    for i, (ticker, _) in enumerate(sorted_tickers[:len(fixed_weights)]):
        weights[ticker] = fixed_weights[i]

    for ticker in predictions:
        if ticker not in weights:
            weights[ticker] = 0.0
    
    return weights

def get_equal_weights(predictions):
    weight = 1.0 / len(predictions)
    return {ticker: weight for ticker in predictions}


def calculate_turnover(old_weights, new_weights):
    if old_weights is None:
        return 0
    return sum(abs(new_weights.get(t, 0) - old_weights.get(t, 0)) 
               for t in set(new_weights) | set(old_weights))

def run_backtest(models, val_dict, feature_cols_dict, fixed_weights, commission=0.0001):
    all_dates = sorted(set.intersection(*[set(df.index) for df in val_dict.values()]))
    
    strategy_returns = []
    benchmark_returns = []
    prev_weights = None
    
    for date in all_dates:
        predictions = {}
        for ticker, model in models.items():
            if date in val_dict[ticker].index:
                X = val_dict[ticker].loc[date, feature_cols_dict[ticker]].values.reshape(1, -1)
                predictions[ticker] = model.predict(X)[0]

        # ===== СТРАТЕГИЯ =====
        new_weights = get_weights_by_rank(predictions, fixed_weights)
        
        # Комиссия
        turnover = calculate_turnover(prev_weights, new_weights)
        commission_cost = turnover * commission
        
        # Доходность стратегии
        ret = sum(new_weights[t] * val_dict[t].loc[date, 'target'] 
                  for t in new_weights if date in val_dict[t].index)
        
        strategy_returns.append(ret - commission_cost)
        prev_weights = new_weights
        
        # ===== БЕНЧМАРК =====
        bench_weights = get_equal_weights(predictions)
        bench_ret = sum(bench_weights[t] * val_dict[t].loc[date, 'target'] 
                        for t in bench_weights if date in val_dict[t].index)
        benchmark_returns.append(bench_ret)
    
    return all_dates[:len(strategy_returns)], strategy_returns, benchmark_returns


def calculate_metrics(returns, name="Стратегия"):
    """Считает и печатает метрики"""
    returns = pd.Series(returns)
    cumprod = (1 + returns).cumprod() - 1
    
    print(f"\n=== {name} ===")
    print(f"Средняя доходность: {returns.mean():.6f}")
    print(f"Волатильность: {returns.std():.6f}")
    print(f"Итоговая доходность: {cumprod.iloc[-1]*100:.2f}%")
    print(f"Sharpe: {returns.mean() / returns.std() * np.sqrt(252):.2f}")
    print(f"Макс. просадка: {returns.min():.2%}")
    
    return cumprod


def plot_results(dates, strategy_cumprod, benchmark_cumprod):
    """Строит график сравнения"""
    plt.figure(figsize=(14, 6))
    plt.plot(dates, strategy_cumprod*100, label='Наша стратегия', linewidth=2)
    plt.plot(dates, benchmark_cumprod*100, label='Равные веса', linewidth=2, linestyle='--')
    plt.title('Сравнение стратегий')
    plt.xlabel('Дата')
    plt.ylabel('Накопленная доходность (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()