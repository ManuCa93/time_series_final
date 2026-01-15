import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from chronos import BaseChronosPipeline
import lightgbm as lgb

# ============================================================================
# BASIC UTILS & BENCHMARK MODELS
# ============================================================================

def cast_df(y, df):
    """Turns a forecast array into a proper DataFrame with timestamps."""
    h = len(y)
    # Starts 1 hour after the last timestamp we have
    return pd.DataFrame(y, index=pd.date_range(start=df.index[-1] + pd.Timedelta(hours=1), periods=h, freq='h'))

def naive(x, h): 
    """Naive forecast: just repeat the last value we saw."""
    return cast_df(np.repeat(x.iloc[-1], h), x)

def mean_fc(x, h): 
    """Mean forecast: use the average of all historical data."""
    return cast_df(np.repeat(x.mean(), h), x)

def seasonal_naive(x, h, m=24):
    """
    Seasonal Naive: repeat the last 24 hours (or whatever m is).
    Good for data with daily patterns.
    """
    values = x.iloc[-m:].values
    # Tile repeats the pattern until we have enough predictions
    y_hat = np.tile(values, int(np.ceil(h/m)))[:h]
    return cast_df(y_hat, x)

def calculate_nmae(y_true, y_pred, y_train_mean):
    """
    Normalized MAE - compares our error to a baseline (predicting the mean).
    Lower is better. < 1 means we beat the baseline.
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mae_baseline = np.mean(np.abs(y_true - np.mean(y_true)))  
    return mae / mae_baseline if mae_baseline != 0 else np.nan

def rolling_window_cv(ts_series, model_func, h_ahead=24, n_splits=10):
    """
    Cross-validation for time series - splits data into chunks to test model stability.
    Can't shuffle time series data, so we use expanding windows instead.
    """
    results = {'mae': [], 'nmae': [], 'actuals': [], 'forecasts': []}
    
    # Figure out how big each step should be
    step = (len(ts_series) - h_ahead) // (n_splits + 1)
    
    for i in range(n_splits):
        # Train on progressively more data each time
        train_end_idx = len(ts_series) - h_ahead - (n_splits - i) * step
        
        train = ts_series.iloc[:train_end_idx]
        test = ts_series.iloc[train_end_idx:train_end_idx + h_ahead]
        
        y_pred = model_func(train, h_ahead).values.ravel()
        y_true = test.values
        
        mae = np.mean(np.abs(y_true - y_pred))
        nmae = calculate_nmae(y_true, y_pred, train.mean())
        
        results['mae'].append(mae)
        results['nmae'].append(nmae)
        results['actuals'].append(y_true)
        results['forecasts'].append(y_pred)
    
    return results

# ============================================================================
# BENCHMARKING SUITES
# ============================================================================

def benchmark_per_rail(ts_data, target_name, split_date='2023-03-15 23:00:00', h_ahead=24):
    """
    Tests all benchmark models on each individual charging station.
    Good for finding which stations are hardest to predict.
    """
    print(f"BENCHMARK NAIVE - {target_name.upper()}")
    
    bench_results = {}
    split_datetime = pd.to_datetime(split_date)
    
    for rail in ts_data.columns:
        ts = ts_data[rail]
        train = ts.loc[:split_date]
        test = ts.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
        
        y_true = test.values
        # Try all three basic methods
        y_naive = naive(train, h_ahead).values.ravel()
        y_mean = mean_fc(train, h_ahead).values.ravel()
        y_snai = seasonal_naive(train, h_ahead).values.ravel()
        
        bench_results[rail] = {
            'Naive': calculate_nmae(y_true, y_naive, train.mean()),
            'Mean': calculate_nmae(y_true, y_mean, train.mean()),
            'S_Naive': calculate_nmae(y_true, y_snai, train.mean())
        }
    
    df_bench = pd.DataFrame(bench_results).T
    
    print(f"Processed {len(bench_results)} rails. Stats:")
    print(df_bench.describe().round(4).loc[['mean', 'std', 'min', 'max']])
    return df_bench


def benchmark_fleet(ts_series, target_name, split_date='2023-03-15 23:00:00', h_ahead=24):
    """
    Tests benchmark models on total fleet consumption/presence.
    Uses both cross-validation and a final test set.
    """
    print(f"BENCHMARK FLEET - {target_name.upper()}")
    
    split_datetime = pd.to_datetime(split_date)
    
    # Cross-validation gives us confidence in model performance
    cv_naive = rolling_window_cv(ts_series, naive, h_ahead=h_ahead, n_splits=10)
    cv_mean = rolling_window_cv(ts_series, mean_fc, h_ahead=h_ahead, n_splits=10)
    cv_snai = rolling_window_cv(ts_series, seasonal_naive, h_ahead=h_ahead, n_splits=10)
    
    print("CV (10-fold) nMAE:")
    print(f"  Naive:   {np.nanmean(cv_naive['nmae']):.4f} +- {np.nanstd(cv_naive['nmae']):.4f}")
    print(f"  Mean:    {np.nanmean(cv_mean['nmae']):.4f} +- {np.nanstd(cv_mean['nmae']):.4f}")
    print(f"  S_Naive: {np.nanmean(cv_snai['nmae']):.4f} +- {np.nanstd(cv_snai['nmae']):.4f}")
    
    # Now test on the actual test period
    train = ts_series.loc[:split_date]
    test = ts_series.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
    
    if len(test) >= h_ahead:
        y_true = test.values
        nmae_naive = calculate_nmae(y_true, naive(train, h_ahead).values.ravel(), train.mean())
        nmae_mean = calculate_nmae(y_true, mean_fc(train, h_ahead).values.ravel(), train.mean())
        nmae_snai = calculate_nmae(y_true, seasonal_naive(train, h_ahead).values.ravel(), train.mean())
        
        print(f"TEST SET ({split_date}) nMAE:")
        print(f"  Naive: {nmae_naive:.4f} | Mean: {nmae_mean:.4f} | S_Naive: {nmae_snai:.4f}")

def plot_fleet_forecast(ts_series, split_date='2023-03-15 23:00:00', h_ahead=24, target_name='Consumption'):
    """Shows how well the benchmark models did visually."""
    
    split_datetime = pd.to_datetime(split_date)
    train = ts_series.loc[:split_date]
    test = ts_series.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
    
    y_naive = naive(train, h_ahead).values.ravel()
    y_mean = mean_fc(train, h_ahead).values.ravel()
    y_snai = seasonal_naive(train, h_ahead).values.ravel()
    
    fig, ax = plt.subplots(figsize=(14, 6))
    hours = np.arange(h_ahead)
    
    ax.plot(hours, test.values, marker='o', label='Actual', linewidth=2.5, 
            color='black', markersize=6, zorder=5)
    ax.plot(hours, y_naive, linestyle='--', label='Naive', linewidth=1.8, alpha=0.8)
    ax.plot(hours, y_mean, linestyle='--', label='Mean', linewidth=1.8, alpha=0.8)
    ax.plot(hours, y_snai, linestyle='--', label='S_Naive', linewidth=1.8, alpha=0.8)
    
    ax.set_title(f'Fleet {target_name}: 24h Ahead Forecast', fontsize=13, fontweight='bold')
    ax.set_xlabel('Hour Ahead', fontsize=11)
    ax.set_ylabel(f'{target_name}', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, h_ahead, 3))
    
    plt.tight_layout()
    return fig

# ============================================================================
# PROBABILISTIC FORECAST (RESIDUAL BASED)
# ============================================================================

def compute_prediction_intervals(ts_series, model_func, split_date='2023-03-15 23:00:00', h_ahead=24, n_splits=10, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
    """
    Figures out how uncertain our predictions are by looking at past errors.
    Builds confidence intervals from historical mistakes.
    """
    split_datetime = pd.to_datetime(split_date)
    all_residuals = []
    
    step = (len(ts_series) - h_ahead) // (n_splits + 1)
    
    # Collect all the errors from past predictions
    for i in range(n_splits):
        train_end_idx = len(ts_series) - h_ahead - (n_splits - i) * step
        train = ts_series.iloc[:train_end_idx]
        test = ts_series.iloc[train_end_idx:train_end_idx + h_ahead]
        
        if len(test) < h_ahead:
            continue
        
        y_pred = model_func(train, h_ahead).values.ravel()
        y_true = test.values
        all_residuals.append(y_true - y_pred)
    
    all_residuals = np.concatenate(all_residuals)
    
    # Find percentiles of the error distribution
    quantile_dict = {}
    for q in quantiles:
        quantile_dict[f'q{int(q*100)}'] = np.quantile(all_residuals, q)
    
    return {
        'residuals': all_residuals,
        'mean_residual': np.mean(all_residuals),
        'std_residual': np.std(all_residuals),
        'quantiles': quantile_dict
    }


def probabilistic_forecast_fleet(ts_series, target_name='Consumption', split_date='2023-03-15 23:00:00', h_ahead=24):
    """
    Makes a forecast with uncertainty bands - not just a single prediction.
    Shows 50%, 80%, and 90% confidence intervals.
    """
    print(f"PROBABILISTIC - {target_name.upper()}")
    
    split_datetime = pd.to_datetime(split_date)
    train = ts_series.loc[:split_date]
    test = ts_series.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
    
    # Get the point forecast
    y_point = seasonal_naive(train, h_ahead).values.ravel()
    
    # Add uncertainty from past errors
    residual_stats = compute_prediction_intervals(
        ts_series, seasonal_naive, split_date, h_ahead, n_splits=10
    )
    
    q = residual_stats['quantiles']
    
    print(f"Residuals | Mean: {residual_stats['mean_residual']:.4f} | Std: {residual_stats['std_residual']:.4f}")
    
    return {
        'point_forecast': y_point,
        'median_pred': y_point + q['q50'],
        'lower_5': y_point + q['q5'],
        'lower_10': y_point + q['q10'],
        'lower_25': y_point + q['q25'],
        'upper_25': y_point + q['q75'],
        'upper_10': y_point + q['q90'],
        'upper_95': y_point + q['q95'],
        'actual': test.values,
        'residual_stats': residual_stats
    }


def plot_probabilistic_forecast(ts_series, target_name='Consumption', split_date='2023-03-15 23:00:00', h_ahead=24):
    """Creates a fan chart - darker = more confident, lighter = less confident."""
    
    fc = probabilistic_forecast_fleet(ts_series, target_name, split_date, h_ahead)
    
    hours = np.arange(h_ahead)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Shaded regions for confidence levels
    ax.fill_between(hours, fc['lower_5'], fc['upper_95'], alpha=0.1, 
                    color='blue', label='90% PI')
    ax.fill_between(hours, fc['lower_10'], fc['upper_10'], alpha=0.15, 
                    color='blue', label='80% PI')
    ax.fill_between(hours, fc['lower_25'], fc['upper_25'], alpha=0.2, 
                    color='blue', label='50% PI')
    
    ax.plot(hours, fc['point_forecast'], color='blue', linewidth=2.5, 
            label='Point Forecast (S_Naive)', zorder=4)
    ax.plot(hours, fc['median_pred'], color='cyan', linewidth=1.5, 
            linestyle=':', label='Median', zorder=3)
    ax.plot(hours, fc['actual'], marker='o', color='black', linewidth=2.5, 
            markersize=6, label='Actual', zorder=5)
    
    ax.set_title(f'{target_name}: Probabilistic 24h Forecast', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, fc


def print_probabilistic_summary(fc_result, h_ahead=24):
    """Checks how often the actual values fell within our predicted ranges."""
    
    print("PROBABILISTIC FORECAST - SUMMARY")
    
    df_summary = pd.DataFrame({
        'Actual': fc_result['actual'],
        'Point': fc_result['point_forecast'],
        'Lo_10': fc_result['lower_10'],
        'Hi_10': fc_result['upper_10']
    })
    
    print(f"First 5 hours:\n{df_summary.head().round(2)}")
    
    # Coverage = how many actuals fell inside the bands
    cov_80 = np.sum((fc_result['actual'] >= fc_result['lower_10']) & (fc_result['actual'] <= fc_result['upper_10']))
    cov_90 = np.sum((fc_result['actual'] >= fc_result['lower_5']) & (fc_result['actual'] <= fc_result['upper_95']))
    
    width_80 = np.mean(fc_result['upper_10'] - fc_result['lower_10'])
    
    print(f"Coverage | 80% PI: {100*cov_80/h_ahead:.1f}% | 90% PI: {100*cov_90/h_ahead:.1f}%")
    print(f"Avg Width (80% PI): {width_80:.2f}")

# ============================================================================
# ML SECTION - LIGHTGBM
# ============================================================================

def make_features(df, lags=[24, 168]):
    """
    Creates features for machine learning models.
    Adds: hour of day, day of week, weekend flag, and lag features.
    """
    df = df.copy()
    if isinstance(df, pd.Series):
        df = df.to_frame(name='target')
    else:
        df.columns = ['target']

    # Time-based features
    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)

    # Lag features (values from 24h and 168h ago)
    for lag in lags:
        df[f'lag_{lag}'] = df['target'].shift(lag)
    
    return df.dropna()

def train_lgbm_fleet(ts_series, target_name='consumption', split_date='2023-03-15 23:00:00', h_ahead=24):
    """
    Trains a gradient boosting model (LightGBM).
    Uses recursive forecasting - predicts one step, then uses that to predict the next.
    """
    print(f"LIGHTGBM - {target_name.upper()}")
    
    df_features = make_features(ts_series, lags=[24, 168])
    
    split_datetime = pd.to_datetime(split_date)
    train_data = df_features.loc[:split_date]
    test_data = df_features.loc[split_datetime + pd.Timedelta(hours=1):]
    
    if len(train_data) == 0:
        return None

    X_train = train_data.drop(columns=['target'])
    y_train = train_data['target']
    
    # Train the model
    model = lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    # Predict step by step (because each prediction depends on the previous one)
    predictions = []
    last_window = ts_series.loc[:split_date].copy()
    current_time = split_datetime + pd.Timedelta(hours=1)
    
    for _ in range(h_ahead):
        # Build features for this timestep
        future_row = pd.DataFrame(index=[current_time])
        future_row['hour'] = current_time.hour
        future_row['dayofweek'] = current_time.dayofweek
        future_row['is_weekend'] = int(current_time.dayofweek in [5, 6])
        
        # Get lag values
        val_lag24 = last_window.loc[current_time - pd.Timedelta(hours=24)] if (current_time - pd.Timedelta(hours=24)) in last_window.index else np.nan
        val_lag168 = last_window.loc[current_time - pd.Timedelta(hours=168)] if (current_time - pd.Timedelta(hours=168)) in last_window.index else np.nan
        
        future_row['lag_24'] = val_lag24
        future_row['lag_168'] = val_lag168
        
        pred_val = model.predict(future_row)[0]
        pred_val = max(0, pred_val)  # Can't have negative consumption
        
        predictions.append(pred_val)
        
        # Add this prediction to history for next step
        last_window = pd.concat([last_window, pd.Series([pred_val], index=[current_time])])
        current_time += pd.Timedelta(hours=1)
    
    y_pred = np.array(predictions)
    mae, nmae = np.nan, np.nan
    
    ground_truth = ts_series.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
    
    if len(ground_truth) == h_ahead:
        y_true = ground_truth.values
        mae = mean_absolute_error(y_true, y_pred)
        nmae = calculate_nmae(y_true, y_pred, y_train.mean())
        print(f"Result | MAE: {mae:.4f} | nMAE: {nmae:.4f}")
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_true': y_true,
        'mae': mae,
        'nmae': nmae,
        'feature_importance': pd.Series(model.feature_importances_, index=X_train.columns)
    }

def plot_lgbm_feature_importance(lgbm_results, target_name='Consumption'):
    """Shows which features the model found most useful."""
    if lgbm_results is None:
        print("No LightGBM results to plot.")
        return

    importance = lgbm_results['feature_importance'].sort_values(ascending=False)
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=importance.values, y=importance.index, palette='viridis', hue=importance.index, legend=False)
    plt.title(f'LightGBM Feature Importance ({target_name})', fontsize=14)
    plt.xlabel('Importance (Split Gain)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    print("Interpretation: 'lag_24' and 'hour' should be most important - confirms daily patterns.")

def plot_lgbm_predictions(lgbm_results, target_name='Consumption'):
    """Plots what LightGBM predicted vs what actually happened."""
    if lgbm_results is None or lgbm_results.get('y_true') is None:
        print("No LightGBM results or ground truth to plot.")
        return

    y_true = lgbm_results['y_true']
    y_pred = lgbm_results['y_pred']
    hours = np.arange(len(y_true))
    
    plt.figure(figsize=(14, 6))
    plt.plot(hours, y_true, marker='o', color='black', label='Actual', linewidth=2.5, zorder=5)
    plt.plot(hours, y_pred, color='green', label='LightGBM', linewidth=2, linestyle='--')
    
    plt.title(f'LightGBM Performance: Actual vs Forecast - {target_name}', fontsize=14, fontweight='bold')
    plt.xlabel('Hours Ahead (Test Set)')
    plt.ylabel(target_name)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ============================================================================
# ML SECTION - CHRONOS 2
# ============================================================================

def train_chronos2_fleet(fleet_ts, target_name='consumption', split_date='2023-03-15 23:00:00', h_ahead=24):
    """
    Uses Chronos 2 - a pre-trained foundation model for time series.
    It's like GPT but for forecasting. No training needed!
    """

    try:
        pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    except Exception as e:
        print(f"Error loading Chronos: {str(e)[:100]}")
        return None
    
    print(f"CHRONOS 2 - {target_name.upper()} | Ctx: {len(fleet_ts.loc[:split_date])} samples")
    
    split_datetime = pd.to_datetime(split_date)
    train = fleet_ts.loc[:split_date]
    test = fleet_ts.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
    
    if len(test) < h_ahead:
        print("Test set too short")
        return None
    
    try:
        # Format data in the way Chronos expects
        df_context = pd.DataFrame({
            'Date Time': train.index,
            'item_id': 'ev_fleet',
            target_name: train.values
        }).reset_index(drop=True)
        
        # Get predictions with uncertainty quantiles
        cron_pred_df = pipeline.predict_df(
            df_context,
            prediction_length=h_ahead,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            id_column="item_id",
            timestamp_column="Date Time",
            target=target_name,
        )
        
        # Use median (0.5) as point forecast
        y_chronos = cron_pred_df['0.5'].values
        y_test = test.values
        mae = mean_absolute_error(y_test, y_chronos)
        nmae = calculate_nmae(y_test, y_chronos, train.mean())
        
        print(f"Result | MAE: {mae:.4f} | nMAE: {nmae:.4f}")
        
        return {
            'pipeline': pipeline,
            'y_pred': y_chronos,
            'y_test': y_test,
            'mae': mae,
            'nmae': nmae,
            'predictions_df': cron_pred_df
        }
    
    except Exception as e:
        print(f"Chronos error: {str(e)[:100]}")
        return None


def train_chronos2_per_rail(ts_data, target_name='consumption', split_date='2023-03-15 23:00:00', h_ahead=24):
    """Runs Chronos on each charging station individually - can take a while!"""
    
    if BaseChronosPipeline is None:
        return None, None

    print(f"CHRONOS 2 PER RAIL - {target_name.upper()}")
    split_datetime = pd.to_datetime(split_date)
    results = {}
    
    try:
        pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    except Exception:
        return None, None
    
    # Loop through all stations
    for rail in tqdm(ts_data.columns, desc='Rails', leave=False):
        ts = ts_data[rail]
        train = ts.loc[:split_date]
        test = ts.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
        
        if len(test) < h_ahead:
            continue
        
        try:
            df_context = pd.DataFrame({
                'Date Time': train.index, 'item_id': rail, target_name: train.values
            }).reset_index(drop=True)
            df_context['Date Time'] = pd.to_datetime(df_context['Date Time'])
            df_context.sort_values('Date Time', inplace=True)
            
            cron_pred_df = pipeline.predict_df(
                df_context, prediction_length=h_ahead, quantile_levels=[0.5],
                id_column="item_id", timestamp_column="Date Time", target=target_name, freq='1h'
            )
            
            y_c = cron_pred_df['0.5'].values
            results[rail] = {
                'MAE': mean_absolute_error(test.values, y_c),
                'nMAE': calculate_nmae(test.values, y_c, train.mean())
            }
        except Exception:
            continue
            
    df_results = pd.DataFrame(results).T
    print(f"Processed {len(df_results)} rails. Stats:")
    if not df_results.empty:
        print(df_results.describe().round(4).loc[['mean', 'std', 'min', 'max']])
    
    return results, df_results


def probabilistic_forecast_chronos2(fleet_ts, target_name='Consumption', split_date='2023-03-15 23:00:00', h_ahead=24):
    """Probabilistic forecast using Chronos - it generates uncertainty natively."""

    print(f"PROB FORECAST (CHRONOS) - {target_name.upper()}")
    split_datetime = pd.to_datetime(split_date)
    train = fleet_ts.loc[:split_date]
    test = fleet_ts.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
    
    try:
        pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        df_context = pd.DataFrame({
            'Date Time': train.index, 'item_id': 'ev_fleet', target_name.lower(): train.values
        }).reset_index(drop=True)
        
        # Request lots of quantiles for a detailed fan chart
        cron_pred_df = pipeline.predict_df(
            df_context, prediction_length=h_ahead, 
            quantile_levels=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95],
            id_column="item_id", timestamp_column="Date Time", target=target_name.lower(),
        )
        
        print("Forecast generated successfully.")
        return {
            'point_forecast': cron_pred_df['0.5'].values,
            'lower_5': cron_pred_df['0.05'].values,
            'lower_10': cron_pred_df['0.1'].values,
            'lower_25': cron_pred_df['0.25'].values,
            'upper_25': cron_pred_df['0.75'].values,
            'upper_10': cron_pred_df['0.9'].values,
            'upper_95': cron_pred_df['0.95'].values,
            'actual': test.values,
            'predictions_df': cron_pred_df
        }
    except Exception as e:
        print(f"Error: {str(e)[:100]}")
        return None
    
def plot_probabilistic_forecast_chronos2(ts_series, target_name='Consumption', split_date='2023-03-15 23:00:00', h_ahead=24):
    """Fan chart for Chronos predictions."""
    
    fc = probabilistic_forecast_chronos2(ts_series, target_name, split_date, h_ahead)
    if fc is None: return None, None
    
    hours = np.arange(h_ahead)
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.fill_between(hours, fc['lower_5'], fc['upper_95'], alpha=0.1, color='blue', label='90% PI')
    ax.fill_between(hours, fc['lower_10'], fc['upper_10'], alpha=0.15, color='blue', label='80% PI')
    ax.fill_between(hours, fc['lower_25'], fc['upper_25'], alpha=0.2, color='blue', label='50% PI')
    
    ax.plot(hours, fc['point_forecast'], color='blue', linewidth=2.5, label='Point (Chronos)', zorder=4)
    ax.plot(hours, fc['actual'], marker='o', color='black', linewidth=2.5, markersize=6, label='Actual', zorder=5)
    
    ax.set_title(f'{target_name}: Probabilistic 24h Forecast - Chronos 2', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig, fc


def print_probabilistic_summary_chronos2(fc_result, h_ahead=24):
    """Prints summary stats for Chronos probabilistic forecast."""
    if fc_result is None: return
    
    print("CHRONOS PROB FORECAST - SUMMARY")
    df_summary = pd.DataFrame({
        'Actual': fc_result['actual'], 'Point': fc_result['point_forecast'],
        'Lo_10': fc_result['lower_10'], 'Hi_10': fc_result['upper_10']
    })
    print(f"First 5 hours:\n{df_summary.head().round(2)}")
    
    # Check how many actual values fell inside the 80% band
    cov_80 = np.sum((fc_result['actual'] >= fc_result['lower_10']) & (fc_result['actual'] <= fc_result['upper_10']))
    print(f"Coverage 80% PI: {100*cov_80/h_ahead:.1f}%")


def compare_all_models_w_chronos2(fleet_ts, target_name='Consumption', split_date='2023-03-15 23:00:00'):
    """
    Battle of the models! Tests S_Naive, ARIMA, LightGBM, and Chronos.
    Shows which one performs best on the test set.
    """
    
    print(f"MODEL COMPARISON - {target_name.upper()}")
    
    split_datetime = pd.to_datetime(split_date)
    train = fleet_ts.loc[:split_date]
    test = fleet_ts.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:24]
    
    results = {}
    
    # Simple baseline
    y_snai = seasonal_naive(train, 24).values.ravel()
    results['S_Naive'] = {'nMAE': calculate_nmae(test.values, y_snai, train.mean())}
    print(f"  S_Naive:  nMAE={results['S_Naive']['nMAE']:.4f}")
    
    # Classic statistical method
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train, order=(2, 1, 2)).fit()
        y_arima = model.forecast(steps=24).values
        results['ARIMA'] = {'nMAE': calculate_nmae(test.values, y_arima, train.mean())}
        print(f"  ARIMA:    nMAE={results['ARIMA']['nMAE']:.4f}")
    except:
        print("  ARIMA:    Failed")
    
    # Gradient boosting
    try:
        lgbm_res = train_lgbm_fleet(fleet_ts, target_name.lower(), split_date)
        if lgbm_res:
             results['LightGBM'] = {'nMAE': lgbm_res['nmae']}
             print(f"  LightGBM: nMAE={results['LightGBM']['nMAE']:.4f}")
    except Exception as e:
        print(f"  LightGBM: Failed ({str(e)[:50]})")

    # Foundation model
    if BaseChronosPipeline:
        c_res = train_chronos2_fleet(fleet_ts, target_name.lower(), split_date)
        if c_res:
            results['Chronos2'] = {'nMAE': c_res['nmae']}
    
    # Declare a winner
    if results:
        df_res = pd.DataFrame(results).T.sort_values('nMAE')
        best = df_res.index[0]
        imp = (results['S_Naive']['nMAE'] - df_res.iloc[0,0]) / results['S_Naive']['nMAE'] * 100
        print(f"BEST: {best} (Imp vs S_Naive: {imp:+.1f}%)")
        return results, df_res
    return {}, pd.DataFrame()


def predict_future_chronos2(ts_data, target_name, h_ahead=24):
    """
    Predicts the actual future (no test set) using Chronos.
    This is what you'd use for real operational forecasting.
    """

    try:
        pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        print(f"FUTURE FORECAST ({h_ahead}h) - {target_name.upper()}")
        
        # Use all available data as context
        context_df = pd.DataFrame({
            'Date Time': pd.to_datetime(ts_data.index), 'item_id': 'ev_fleet', target_name: ts_data.values
        }).reset_index(drop=True)
        
        # Predict into the unknown future
        future_predictions_df = pipeline.predict_df(
            context_df, prediction_length=h_ahead, quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            id_column="item_id", timestamp_column="Date Time", target=target_name
        )
        
        # Give it proper future timestamps
        future_predictions_df.index = pd.date_range(ts_data.index[-1] + pd.Timedelta(hours=1), periods=h_ahead, freq='h')
        print(future_predictions_df[['0.5']].rename(columns={'0.5': 'Median'}).head())
        return future_predictions_df

    except Exception as e:
        print(f"Error: {str(e)[:100]}")
        return None

def plot_future_forecast(ts_data, forecast_df, target_name):
    """Shows the historical data and extends it with future predictions."""
    plt.figure(figsize=(15, 7))
    plt.plot(ts_data.index, ts_data.values, label='History', color='black', alpha=0.8)
    
    # Add confidence bands if they exist
    if '0.1' in forecast_df.columns:
        plt.fill_between(forecast_df.index, forecast_df['0.1'], forecast_df['0.9'], color='skyblue', alpha=0.4, label='80% PI')
    if '0.25' in forecast_df.columns:
        plt.fill_between(forecast_df.index, forecast_df['0.25'], forecast_df['0.75'], color='steelblue', alpha=0.6, label='50% PI')
    if '0.5' in forecast_df.columns:
        plt.plot(forecast_df.index, forecast_df['0.5'], label='Median', color='red', linestyle='--', linewidth=2)
    
    plt.title(f'Future Forecast: {target_name}', fontsize=16, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()

def predict_future_snaive(ts_data, target_name, h_ahead=24):
    """
    Simple future forecast - just repeats the last 24 hours.
    Good baseline for presence data where patterns are very stable.
    """
    print(f"FUTURE FORECAST (S_NAIVE) - {target_name.upper()}")
    
    # Grab last day
    last_cycle = ts_data.iloc[-24:].values
    
    # Repeat it as many times as needed
    y_pred = np.tile(last_cycle, int(np.ceil(h_ahead/24)))[:h_ahead]
    
    future_idx = pd.date_range(start=ts_data.index[-1] + pd.Timedelta(hours=1), periods=h_ahead, freq='h')
    
    df_future = pd.DataFrame(index=future_idx)
    df_future['0.5'] = y_pred  # Point forecast
    df_future['0.1'] = y_pred  # Fake lower bound (no uncertainty here)
    df_future['0.9'] = y_pred  # Fake upper bound
    
    print(f"S_Naive Forecast Generated. First 5 values:\n{df_future.head()}")
    return df_future

def print_forecast_report(forecast_df, target_name, model_name="Chronos"):
    """
    Generates a nice summary report that you could show to a grid operator.
    Gives totals, ranges, and actionable interpretation.
    """
    
    total_median = forecast_df['0.5'].sum()
    total_low = forecast_df['0.1'].sum()
    total_high = forecast_df['0.9'].sum()
    
    print("\n" + "="*60)
    print(f"OPERATIONAL REPORT: NEXT 24H - {target_name.upper()} ({model_name})")
    print("="*60)
    
    if target_name.lower() == 'consumption':
        print(f"Expected Total Load:   {total_median:.2f} kWh")
        if model_name == "Chronos":
            print(f"Probabilistic Range:   [{total_low:.2f} kWh  -  {total_high:.2f} kWh] (80% Conf.)")
            print("-" * 60)
            print("INTERPRETATION FOR GRID OPERATOR:")
            print(f"• Plan for {total_median:.0f} kWh. Reserve capacity for peak of {total_high:.0f} kWh.")
        else:
            print("(Deterministic Forecast - No Uncertainty Range)")
            
    elif target_name.lower() == 'presence':
        peak_cars = forecast_df['0.5'].max()
        print(f"Expected Peak Occupancy: {peak_cars:.1f} vehicles")
        print("-" * 60)
        print("INTERPRETATION:")
        print(f"• Expect parking to reach max occupancy of around {int(peak_cars)} cars.")
        
    print("="*60 + "\n")