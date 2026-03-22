import time

import yfinance as yf
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
import json
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / 'models'
threshold_path = MODELS_DIR / 'threshold.json'
catboost_regressor_path = MODELS_DIR / 'catboost_regressor.cbm'
catboost_classifier_path = MODELS_DIR / 'catboost_classifier.cbm'

with open(threshold_path) as f:
    threshold = json.load(f)['volatility_threshold']


def get_predictions():
    try:
        tickers = ['BTC-USD', 'XRP-USD', 'DOGE-USD', 'SOL-USD', 'ETH-USD']
        for attempt in range(3):
            df = yf.download(tickers, period='60d')
            if df is not None and not df.empty:
                break
            time.sleep(3)
        else:
            return None
        df_long = df[["Close", "Volume", "Open", "High", "Low"]].stack(future_stack=True).reset_index()
        df_long = df_long.sort_values(["Ticker", "Date"])
        df_long['return'] = df_long.groupby('Ticker')['Close'].pct_change()
        df_long['log_return'] = np.log(1 + df_long['return'])
        df_long['lag_1'] = df_long.groupby('Ticker')['log_return'].shift(1)
        df_long['lag_7'] = df_long.groupby('Ticker')['log_return'].shift(7)
        df_long['lag_14'] = df_long.groupby('Ticker')['log_return'].shift(14)
        df_long['rolling_mean_7'] = df_long.groupby('Ticker')['log_return'].transform(lambda x: x.shift(1).rolling(7).mean())
        df_long['rolling_std_7'] = df_long.groupby('Ticker')['log_return'].transform(lambda x: x.shift(1).rolling(7).std())
        df_long['price_change_pct'] = df_long.groupby('Ticker')['Close'].transform(lambda x: x.shift(1).pct_change(7))
        #временные фичи
        df_long['day_of_week'] = df_long["Date"].dt.dayofweek
        df_long['month'] = df_long['Date'].dt.month
        # фичи волатильности(аномально высокой)
        df_long['high_volatility'] = (df_long['rolling_std_7'] > threshold).astype(int)
        df_long = df_long.dropna()

        df_today = df_long.groupby('Ticker').last().reset_index()

        X_today_regr = df_today.drop(['Date', 'log_return','return', 'Close',
                                        'Open', 'High', 'Low', 'high_volatility'], axis=1)
        X_today_class= df_today.drop(['Date', 'log_return','return', 'Close', 'Open',
                                         'High', 'Low', 'high_volatility', 'rolling_std_7'], axis=1)

        model_regr = CatBoostRegressor(cat_features=['Ticker'])
        model_class = CatBoostClassifier(cat_features=['Ticker'])

        model_regr.load_model(catboost_regressor_path)
        model_class.load_model(catboost_classifier_path)

        predictions_regr = model_regr.predict(X_today_regr)
        predictions_class = model_class.predict(X_today_class)

        proba = model_class.predict_proba(X_today_class)[:, 1]

        df_final = pd.DataFrame({'Ticker': df_today['Ticker'].values,
                                'predicted price change': predictions_regr,
                                'probability of high volatility': proba})

        df_final['predicted price change'] = (np.exp(df_final['predicted price change']) - 1) * 100
        sorted_df = df_final.sort_values(['predicted price change', 'probability of high volatility'],ascending=[False, True])
    except Exception as e:
        print(f'ошибка {e}')
        return  None
    return sorted_df

if __name__ == '__main__':
    result = get_predictions()
    if result is not None:
        print(result)