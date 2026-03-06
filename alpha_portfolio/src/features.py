import pandas as pd
import numpy as np

from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def prepare_features_per_asset(stocks_with_alphas, lookbacks=[1, 2, 3, 5, 10]):
    # колонки с альфами определим по названию, чтобы можно было менять их количество
    first_ticker = list(stocks_with_alphas.keys())[0]
    alpha_cols = [col for col in stocks_with_alphas[first_ticker].columns 
                  if col.startswith('alpha')]

    features_per_asset = {}
    
    for ticker, df in stocks_with_alphas.items():
        df_features = df.copy()

        df_features[alpha_cols] = df_features[alpha_cols].apply(pd.to_numeric, errors='coerce')

        for col in alpha_cols:
            df_features[col] = df_features[col].replace([np.inf, -np.inf], np.nan)

        df_features['target'] = df_features['Return'].shift(-1)

        
        for col in alpha_cols:
            for lag in lookbacks:
                df_features[f'{col}_lag{lag}'] = df_features[col].shift(lag)

        df_features['return_lag1'] = df_features['Return'].shift(1)
        
        df_features = df_features.dropna()
        
        features_per_asset[ticker] = df_features
            
    return features_per_asset

def split_data_per_asset(features_per_asset, train_ratio=0.6, val_ratio=0.35):
    train_dict = {}
    val_dict = {}
    test_dict = {}
    
    for ticker, df in features_per_asset.items():
        dates = sorted(df.index)
        n = len(dates)
        # Разделим данные срезами по индексам
        train_idx = int(n * train_ratio)
        val_idx = int(n * (train_ratio + val_ratio))

        train_dates = dates[:train_idx]
        val_dates = dates[train_idx:val_idx]
        test_dates = dates[val_idx:]
        
        train_dict[ticker] = df.loc[train_dates]
        val_dict[ticker] = df.loc[val_dates]
        test_dict[ticker] = df.loc[test_dates]
            
    return train_dict, val_dict, test_dict


def train_model_for_ticker(train_dict, ticker, model_type='linear', use_pca=False, n_components=20, alpha=1.0):

    df_train = train_dict[ticker]
    
    # Фичи: все, кроме сырых цен и target
    feature_cols = [col for col in df_train.columns 
                    if col not in ['Open', 'Low', 'High', 'Close', 'Avg', 'Volume', 'Count', 'target']]
    
    X_train = df_train[feature_cols]
    y_train = df_train['target']
    
    if len(X_train) == 0:
        print(f"  ПРЕДУПРЕЖДЕНИЕ: {ticker} нет тренировочных данных")
        return None, None
    
    # Выбираем модель
    if model_type == 'ridge':
        base_model = Ridge(alpha=alpha)
    elif model_type == 'lasso':
        base_model = Lasso(alpha=alpha, max_iter=10000)
    else:  # linear
        base_model = LinearRegression()
    
    # Строим пайплайн
    if use_pca:
        model = Pipeline([
            ('scaler', StandardScaler()),  # Нормализация перед PCA
            ('pca', PCA(n_components=min(n_components, len(feature_cols)))),
            ('regressor', base_model)
        ])
    else:
        model = Pipeline([
            ('scaler', StandardScaler()),  # Нормализация для устойчивости
            ('regressor', base_model)
        ])
    
    model.fit(X_train, y_train)
    
    return model, feature_cols


def train_all_models(train_dict, model_type='linear', use_pca=False, n_components=20, alpha=1.0):
    """
    Обучает модели для всех тикеров
    """
    models = {}
    features = {}
    
    for ticker in train_dict.keys():
        print(f"Обучение {ticker} (model={model_type}, pca={use_pca})...")
        model, feature_cols = train_model_for_ticker(
            train_dict, ticker, 
            model_type=model_type, 
            use_pca=use_pca, 
            n_components=n_components, 
            alpha=alpha
        )
        if model is not None:
            models[ticker] = model
            features[ticker] = feature_cols
    
    return models, features


