import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Добавляем путь к src, чтобы импортировать alphas
sys.path.append(str(Path(__file__).parent))
from alphas import Alphas, get_alpha

def calculate_alphas_for_stock(df, ticker):
    df = df.copy()
    alpha_calc = Alphas(df)
    df_with_alphas = get_alpha(df)
    # Оставляем только те колонки, которые начинаются с 'alpha'
    alpha_columns = [col for col in df_with_alphas.columns if col.startswith('alpha')]
    
    result = df.copy()
    for col in alpha_columns:
        result[col] = df_with_alphas[col]
    
    return result

def calculate_all_alphas(stocks_dict):
    stocks_with_alphas = {}
    
    for ticker, df in stocks_dict.items():
        df_with_alphas = calculate_alphas_for_stock(df, ticker)
        stocks_with_alphas[ticker] = df_with_alphas
    
    return stocks_with_alphas

