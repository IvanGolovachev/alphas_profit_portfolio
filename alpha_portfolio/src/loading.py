import pandas as pd
import numpy as np
from pathlib import Path

def load_stock_data(file_path):
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.sort_index()
    
    # Очистим колонки от пробелов
    for col in ['Open', 'High', 'Low', 'Close', 'Volume', 'Avg']:
        df[col] = df[col].astype(str).str.replace(r'\s+', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['Return'] = df['Close'].diff() / df['Close']
    
    return df


def load_all_stocks(data_folder): 
    data_folder = Path(data_folder)
    stock_data = {}
    
    # Сначала загружаем все данные и собираем даты
    all_dfs = {}
    all_dates_sets = []
    
    for file_path in data_folder.glob('*.csv'):
        ticker = file_path.stem
        df = load_stock_data(file_path)
        all_dfs[ticker] = df
        all_dates_sets.append(set(df.index))  # Превращаем индекс в множество
    
    # Находим общие даты для всех файлов
    common_dates = set.intersection(*all_dates_sets)
    common_dates = sorted(list(common_dates))
    print(f"Общих дат для всех акций: {len(common_dates)}")
    
    # Обрезаем каждую акцию до общих дат
    for ticker, df in all_dfs.items():
        stock_data[ticker] = df.loc[common_dates]
    
    return stock_data
