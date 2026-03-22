import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import pandas as pd
    import yfinance as yf
    import matplotlib.pyplot as plt
    import numpy as np
    from catboost import CatBoostRegressor, CatBoostClassifier 
    from sklearn.metrics import  mean_absolute_error, root_mean_squared_error, mean_squared_error , accuracy_score, f1_score
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    import mlflow
    import mlflow.catboost
    import shap
    import optuna
    from sklearn.dummy import DummyClassifier
    import joblib

    return (
        CatBoostClassifier,
        CatBoostRegressor,
        DummyClassifier,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        mean_absolute_error,
        mlflow,
        np,
        optuna,
        plt,
        root_mean_squared_error,
        shap,
        sns,
        yf,
    )


@app.cell
def _(yf):
    tickers = ['BTC-USD', 'XRP-USD', 'DOGE-USD', 'SOL-USD', 'ETH-USD']
    df = yf.download(tickers, start='2023-01-01', end='2026-01-01')
    return (df,)


@app.cell
def _(df):
    df.head()
    return


@app.cell
def _(df, np):
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
    treshold = df_long['rolling_std_7'].quantile(0.75)
    df_long['high_volatility'] = (df_long['rolling_std_7'] > treshold).astype(int)

    df_long = df_long.dropna()
    df_long.head(10)
    return (df_long,)


@app.cell
def _(df_long):
    df_long['log_return'].describe()
    return


@app.cell
def _(df_long):
    df_long.isna().sum()
    return


@app.cell
def _(CatBoostRegressor, df_long):
    model_regr = CatBoostRegressor(cat_features=['Ticker'])

    train_data = df_long[df_long['Date'] < '2025-06-01']
    val_data = df_long[df_long['Date'] >= '2025-06-01']

    X_train_regr = train_data.drop(['Date', 'log_return','return', 'Close',
                                    'Open', 'High', 'Low', 'high_volatility'], axis=1)

    X_val_regr = val_data.drop(['Date', 'log_return','return', 'Close',
                                'Open', 'High', 'Low','high_volatility'], axis=1)

    y_train_regr = train_data['log_return']
    y_val_regr = val_data['log_return']
    return (
        X_train_regr,
        X_val_regr,
        model_regr,
        train_data,
        val_data,
        y_train_regr,
        y_val_regr,
    )


@app.cell
def _(X_train_regr, model_regr, y_train_regr):
    model_regr.fit(X_train_regr, y_train_regr)
    return


@app.cell
def _(
    X_train_regr,
    X_val_regr,
    mean_absolute_error,
    model_regr,
    root_mean_squared_error,
    y_train_regr,
    y_val_regr,
):
    y_pred_train_regr = model_regr.predict(X_train_regr)
    y_pred_val_regr = model_regr.predict(X_val_regr)

    train_mae = mean_absolute_error(y_train_regr, y_pred_train_regr)
    train_rmse = root_mean_squared_error(y_train_regr, y_pred_train_regr)
    val_mae = mean_absolute_error(y_val_regr, y_pred_val_regr)
    val_rmse = root_mean_squared_error(y_val_regr, y_pred_val_regr)

    print(f"train MAE: {train_mae:.4f}")
    print(f"test MAE: {val_mae:.4f}")
    print(f"train RMSE: {train_rmse:.4f}")
    print(f"test RMSE: {val_rmse:.4f}")
    return val_mae, val_rmse


@app.cell
def _(model_regr):
    model_regr.get_feature_importance(prettified=True)
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


@app.cell
def _(CatBoostClassifier, train_data, val_data):
    model_class = CatBoostClassifier(cat_features=['Ticker'])

    X_train_class = train_data.drop(['Date', 'log_return','return', 'Close', 'Open',
                                     'High', 'Low', 'high_volatility', 'rolling_std_7'], axis=1)
    X_val_class = val_data.drop(['Date', 'log_return','return', 'Close', 'Open',
                                 'High', 'Low', 'high_volatility', 'rolling_std_7'], axis=1)

    y_train_class =  train_data['high_volatility']
    y_val_class = val_data['high_volatility']
    return X_train_class, X_val_class, model_class, y_train_class, y_val_class


@app.cell
def _(X_train_class, model_class, y_train_class):
    model_class.fit(X_train_class, y_train_class)
    return


@app.cell
def _(
    X_train_class,
    X_val_class,
    accuracy_score,
    f1_score,
    model_class,
    y_train_class,
    y_val_class,
):
    y_pred_train_class = model_class.predict(X_train_class)
    y_pred_val_class = model_class.predict(X_val_class)

    train_acc = accuracy_score(y_train_class, y_pred_train_class)
    train_f1 = f1_score(y_train_class, y_pred_train_class)
    val_acc = accuracy_score(y_val_class, y_pred_val_class)
    val_f1 = f1_score(y_val_class, y_pred_val_class)

    print(f"train acc: {train_acc:.4f}")
    print(f"test acc: {val_acc:.4f}")
    print(f"train f1: {train_f1:.4f}")
    print(f"test f1: {val_f1:.4f}")
    return val_acc, val_f1, y_pred_val_class


@app.cell
def _(classification_report, y_pred_val_class, y_val_class):
    print(classification_report(y_val_class, y_pred_val_class))
    return


@app.cell
def _(model_class):
    model_class.get_feature_importance(prettified=True)
    return


@app.cell
def _(df_long):
    df_long['high_volatility'].value_counts()
    return


@app.cell
def _(confusion_matrix, plt, sns, y_pred_val_class, y_val_class):
    matrix = confusion_matrix(y_true=y_val_class, y_pred=y_pred_val_class)
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Greens', annot_kws={'fontsize': 14})
    plt.ylabel('Истинные значения')
    plt.xlabel('Предсказанные значения')
    plt.title('Матрица ошибок', pad=15)
    plt.show()
    return


@app.cell
def _(mean_absolute_error, val_data, val_mae, y_val_regr):
    naive_pred = val_data['lag_1']  # просто предыдущий день
    naive_mae  = mean_absolute_error(y_val_regr.dropna(), naive_pred.dropna())
    #naive_acc = accuracy_score(y_val_class, naive_pred)
    print(f"Naive MAE: {naive_mae:.4f}, val mae: {val_mae}")
    #print(f"Naive acc: {naive_acc:.4f}")

    #Если `naive_mae` ≈ твоему `val_mae` — модель не научилась ничему сверх "смотри на вчера". Это нормально для baseline, но важно знать.
    return (naive_mae,)


@app.cell
def _(X_val_regr, model_regr, shap):
    explainer = shap.TreeExplainer(model_regr)
    shap_values = explainer(X_val_regr)
    shap.summary_plot(shap_values, X_val_regr)
    return


@app.cell
def _(
    DummyClassifier,
    X_train_class,
    X_val_class,
    accuracy_score,
    f1_score,
    y_train_class,
    y_val_class,
):
    dummy = DummyClassifier(strategy='most_frequent')
    dummy.fit(X_train_class, y_train_class)
    y_pred_dummy = dummy.predict(X_val_class)
    acc_naive = accuracy_score(y_val_class, y_pred_dummy)
    f1_naive = f1_score(y_val_class, y_pred_dummy)
    return acc_naive, f1_naive


@app.cell
def _(
    acc_naive,
    f1_naive,
    mlflow,
    model_class,
    model_regr,
    naive_mae,
    val_acc,
    val_f1,
    val_mae,
    val_rmse,
):
    mlflow.set_experiment('crypto_baseline')

    with mlflow.start_run(run_name='catboost_regressor'):
        params = model_regr.get_params()
        mlflow.log_params(params)
        # Метрики
        mlflow.log_metric("val_mae", val_mae)
        mlflow.log_metric("val_rmse", val_rmse)
        mlflow.log_metric('naive_model', naive_mae)
        mlflow.log_metric('improvement_pct', (naive_mae - val_mae) / naive_mae * 100)
        # Сама модель
        mlflow.catboost.log_model(model_regr, "model")

    with mlflow.start_run(run_name='catboost_classifier'):
        params = model_class.get_params()
        mlflow.log_params(params)
        mlflow.log_metric('f1', val_f1)
        mlflow.log_metric('acc', val_acc)
        mlflow.log_metric('naive_acc', acc_naive)
        mlflow.log_metric('naive_f1', f1_naive)
        mlflow.catboost.log_model(model_class, 'model_class')
    return


@app.cell
def _(
    CatBoostRegressor,
    X_train_regr,
    X_val_regr,
    mean_absolute_error,
    mlflow,
    y_train_regr,
    y_val_regr,
):
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

    return (objective_regr,)


@app.cell
def _(
    CatBoostClassifier,
    X_train_class,
    X_val_class,
    f1_score,
    mlflow,
    y_train_class,
    y_val_class,
):
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

    return


@app.cell
def _(mlflow, objective_regr, optuna):
    mlflow.set_experiment('optuna')
    with mlflow.start_run(run_name="optuna_search"):
        study = optuna.create_study(direction='minimize')
        study.optimize(objective_regr, n_trials=15)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_mae", study.best_value)
    return (study,)


@app.cell
def _(study):
    print("Лучшие параметры:", study.best_params)
    print("Лучший val_mae:", study.best_value)
    return


@app.cell
def _(CatBoostRegressor, X_train_regr, study, y_train_regr):
    best_model_regr = CatBoostRegressor(**study.best_params, cat_features=['Ticker'])
    best_model_regr.fit(X_train_regr, y_train_regr)
    return (best_model_regr,)


@app.cell
def _(
    X_val_regr,
    best_model_regr,
    mean_absolute_error,
    root_mean_squared_error,
    y_val_regr,
):
    y_pred_best_regr = best_model_regr.predict(X_val_regr)
    mae_best = mean_absolute_error(y_val_regr,y_pred_best_regr)
    rmse_best = root_mean_squared_error(y_val_regr,y_pred_best_regr)
    print(f"best MAE: {mae_best:.4f}")
    print(f"best RMSE: {rmse_best:.4f}")
    return


@app.cell
def _(best_model_regr):
    best_model_regr.save_model('models/catboost_regressor.cbm')
    return


@app.cell
def _():
    import os
    print(os.getcwd())
    return


@app.cell
def _(df_long):
    df_today = df_long.groupby('Ticker').last().reset_index()
    X_today_regr = df_today.drop(['Date', 'log_return','return', 'Close',
                                    'Open', 'High', 'Low', 'high_volatility'], axis=1)

    return (df_today,)


@app.cell
def _(df_today):
    df_today
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
