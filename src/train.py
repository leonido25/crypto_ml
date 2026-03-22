from catboost import CatBoostRegressor
import yfinance as yf
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, CatBoostClassifier
import mlflow
import mlflow.catboost
import optuna
from sklearn.dummy import DummyClassifier
from sklearn.metrics import mean_absolute_error, f1_score
import json
from pathlib import Path
tickers = ['BTC-USD', 'XRP-USD', 'DOGE-USD', 'SOL-USD', 'ETH-USD']
df = yf.download(tickers, start='2023-01-01', end='2026-01-01')

df_long = df[["Close", "Volume", "Open", "High", "Low"]].stack(level=1).reset_index()
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

threshold = df_long['rolling_std_7'].quantile(0.75)
MODELS_DIR = Path(__file__).parent.parent / 'models'
threshold_path = MODELS_DIR / 'threshold.json'

with open(threshold_path, 'w') as f:
    json.dump({'volatility_threshold': float(threshold)}, f)

df_long['high_volatility'] = (df_long['rolling_std_7'] > threshold).astype(int)
df_long = df_long.dropna()
# обучение моделей

model_regr = CatBoostRegressor(cat_features=['Ticker'])
train_data = df_long[df_long['Date'] < '2025-06-01']
val_data = df_long[df_long['Date'] >= '2025-06-01']
X_train_regr = train_data.drop(['Date', 'log_return','return', 'Close',
                                'Open', 'High', 'Low', 'high_volatility'], axis=1)
X_val_regr = val_data.drop(['Date', 'log_return','return', 'Close',
                            'Open', 'High', 'Low','high_volatility'], axis=1)
y_train_regr = train_data['log_return']
y_val_regr = val_data['log_return']

#model_regr.fit(X_train_regr, y_train_regr)

# classifier
model_class = CatBoostClassifier(cat_features=['Ticker'])
X_train_class = train_data.drop(['Date', 'log_return','return', 'Close', 'Open',
                                 'High', 'Low', 'high_volatility', 'rolling_std_7'], axis=1)
X_val_class = val_data.drop(['Date', 'log_return','return', 'Close', 'Open',
                             'High', 'Low', 'high_volatility', 'rolling_std_7'], axis=1)
y_train_class =  train_data['high_volatility']
y_val_class = val_data['high_volatility']

#model_class.fit(X_train_class, y_train_class)

def objective_regr(trial):
    with mlflow.start_run(nested=True):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        }
        mlflow.log_params(params)
        model = CatBoostRegressor(**params, cat_features=['Ticker'], verbose=0)
        model.fit(X_train_regr, y_train_regr)
        y_pred = model.predict(X_val_regr)
        mae = mean_absolute_error(y_val_regr, y_pred)
        mlflow.log_metric('mae', mae)
    return mae

def objective_class(trial):
    with mlflow.start_run(nested=True):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'depth': trial.suggest_int('depth', 4, 10),
            'iterations': trial.suggest_int('iterations', 100, 1000),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        }
        mlflow.log_params(params)
        model = CatBoostClassifier(**params, cat_features=['Ticker'], verbose=0)
        model.fit(X_train_class, y_train_class)
        y_pred = model.predict(X_val_class)
        f1 = f1_score(y_val_class, y_pred)
        mlflow.log_metric('f1', f1)
    return f1

mlflow.set_experiment('optuna')
with mlflow.start_run(run_name="optuna_search_regression"):
    study_r = optuna.create_study(direction='minimize')
    study_r.optimize(objective_regr, n_trials=15)
    mlflow.log_params(study_r.best_params)
    mlflow.log_metric("best_val_mae", study_r.best_value)

print("Лучшие параметры регрессии:", study_r.best_params)
print("Лучший val_mae:", study_r.best_value)

with mlflow.start_run(run_name="optuna_search_classification"):
    study_c = optuna.create_study(direction='maximize')
    study_c.optimize(objective_class, n_trials=15)
    mlflow.log_params(study_c.best_params)
    mlflow.log_metric("best_f1", study_c.best_value)

print("Лучшие параметры классификации:", study_c.best_params)
print("Лучший f1:", study_c.best_value)

best_model_regr = CatBoostRegressor(**study_r.best_params, cat_features=['Ticker'])
best_model_regr.fit(X_train_regr, y_train_regr)

regr_model_path = MODELS_DIR / 'catboost_regressor.cbm'
class_model_path = MODELS_DIR / 'catboost_classifier.cbm'

best_model_regr.save_model(regr_model_path)

best_model_class = CatBoostClassifier(**study_c.best_params, cat_features=['Ticker'])
best_model_class.fit(X_train_class, y_train_class)
best_model_class.save_model(class_model_path)
