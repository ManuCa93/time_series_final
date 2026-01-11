import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def cast_df(y, df):
    h = len(y)
    return pd.DataFrame(y, index=pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1), periods=h, freq='h'))

def naive(x, h): 
    return cast_df(np.repeat(x.iloc[-1], h), x)

def mean_fc(x, h): 
    return cast_df(np.repeat(x.mean(), h), x)

def seasonal_naive(x, h, m=24):
    values = x.iloc[-m:].values
    y_hat = np.tile(values, int(np.ceil(h/m)))[:h]
    return cast_df(y_hat, x)

def calculate_nmae(y_true, y_pred, y_train_mean):
    # mae = np.mean(np.abs(y_true - y_pred))
    # mae_baseline = np.mean(np.abs(y_true - y_train_mean))
    # return mae / mae_baseline if mae_baseline != 0 else np.nan
    mae = np.mean(np.abs(y_true - y_pred))
    mae_baseline = np.mean(np.abs(y_true - np.mean(y_true)))  
    return mae / mae_baseline if mae_baseline != 0 else np.nan

def rolling_window_cv(ts_series, model_func, h_ahead=24, n_splits=10):
    """ time series CV w expanding window"""
    
    results = {'mae': [], 'nmae': [], 'actuals': [], 'forecasts': []}
    
    step = (len(ts_series) - h_ahead) // (n_splits + 1)
    
    for i in range(n_splits):
        train_end_idx = len(ts_series) - h_ahead - (n_splits - i) * step
        train = ts_series.iloc[:train_end_idx]
        test = ts_series.iloc[train_end_idx:train_end_idx + h_ahead]
        
        if len(test) < h_ahead:
            continue
        
        y_pred = model_func(train, h_ahead).values.ravel()
        y_true = test.values
        
        mae = np.mean(np.abs(y_true - y_pred))
        nmae = calculate_nmae(y_true, y_pred, train.mean())
        
        results['mae'].append(mae)
        results['nmae'].append(nmae)
        results['actuals'].append(y_true)
        results['forecasts'].append(y_pred)
    
    return results

# Benchmark for rails
def benchmark_per_rail(ts_data, target_name, split_date='2023-03-15 23:00:00', h_ahead=24):
    """Naive results for each rail"""
    
    print(f"\n{'='*55}")
    print(f"BENCHMARK NAIVE - {target_name.upper()}")
    print(f"{'='*55}")
    
    bench_results = {}
    split_datetime = pd.to_datetime(split_date)
    
    for rail in ts_data.columns:
        ts = ts_data[rail]
        train = ts.loc[:split_date]
        test = ts.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
        
        if len(test) < h_ahead:
            continue
        
        y_true = test.values
        
        # Forecasts
        y_naive = naive(train, h_ahead).values.ravel()
        y_mean = mean_fc(train, h_ahead).values.ravel()
        y_snai = seasonal_naive(train, h_ahead).values.ravel()
        
        # Metrics
        bench_results[rail] = {
            'Naive': calculate_nmae(y_true, y_naive, train.mean()),
            'Mean': calculate_nmae(y_true, y_mean, train.mean()),
            'S_Naive': calculate_nmae(y_true, y_snai, train.mean())
        }
    
    df_bench = pd.DataFrame(bench_results).T
    
    print(f"\nProcessed {len(bench_results)} rails\n")
    print(df_bench.round(4))
    print(f"\n{'-'*55}")
    print("Summary Statistics:")
    print(df_bench.describe().round(4))
    print(f"{'-'*55}\n")
    
    return df_bench


# Benchmark fleet
def benchmark_fleet(ts_series, target_name, split_date='2023-03-15 23:00:00', h_ahead=24):
    """Benchmark fleet level con rolling window CV + test set."""
    
    print(f"\n{'='*55}")
    print(f"BENCHMARK FLEET - {target_name.upper()}")
    print(f"{'='*55}\n")
    
    split_datetime = pd.to_datetime(split_date)
    
    # Cross-validation
    print("📊 ROLLING WINDOW CV (10 folds):")
    print(f"{'-'*55}")
    
    cv_naive = rolling_window_cv(ts_series, naive, h_ahead=h_ahead, n_splits=10)
    cv_mean = rolling_window_cv(ts_series, mean_fc, h_ahead=h_ahead, n_splits=10)
    cv_snai = rolling_window_cv(ts_series, seasonal_naive, h_ahead=h_ahead, n_splits=10)
    
    print(f"\nNaive   | MAE: {np.mean(cv_naive['mae']):.4f} ± {np.std(cv_naive['mae']):.4f} | "
          f"nMAE: {np.nanmean(cv_naive['nmae']):.4f} ± {np.nanstd(cv_naive['nmae']):.4f}")
    print(f"Mean    | MAE: {np.mean(cv_mean['mae']):.4f} ± {np.std(cv_mean['mae']):.4f} | "
          f"nMAE: {np.nanmean(cv_mean['nmae']):.4f} ± {np.nanstd(cv_mean['nmae']):.4f}")
    print(f"S_Naive | MAE: {np.mean(cv_snai['mae']):.4f} ± {np.std(cv_snai['mae']):.4f} | "
          f"nMAE: {np.nanmean(cv_snai['nmae']):.4f} ± {np.nanstd(cv_snai['nmae']):.4f}")
    
    # Test set
    print(f"\n{'-'*55}")
    print(f"📋 TEST SET ({split_date}):")
    print(f"{'-'*55}\n")
    
    train = ts_series.loc[:split_date]
    test = ts_series.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
    
    if len(test) >= h_ahead:
        y_true = test.values
        
        y_naive = naive(train, h_ahead).values.ravel()
        y_mean = mean_fc(train, h_ahead).values.ravel()
        y_snai = seasonal_naive(train, h_ahead).values.ravel()
        
        nmae_naive = calculate_nmae(y_true, y_naive, train.mean())
        nmae_mean = calculate_nmae(y_true, y_mean, train.mean())
        nmae_snai = calculate_nmae(y_true, y_snai, train.mean())
        
        print(f"Naive   nMAE: {nmae_naive:.4f}")
        print(f"Mean    nMAE: {nmae_mean:.4f}")
        print(f"S_Naive nMAE: {nmae_snai:.4f} ← BASELINE\n")


def plot_fleet_forecast(ts_series, split_date='2023-03-15 23:00:00', h_ahead=24, target_name='Consumption'):
    """Plot fleet forecast vs actual."""
    
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


# 6. PROBABILISTIC FORECAST
def compute_prediction_intervals(ts_series, model_func, split_date='2023-03-15 23:00:00', 
                                 h_ahead=24, n_splits=10, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
    """Prediction intervals from residuals"""
    
    split_datetime = pd.to_datetime(split_date)
    all_residuals = []
    
    step = (len(ts_series) - h_ahead) // (n_splits + 1)
    
    for i in range(n_splits):
        train_end_idx = len(ts_series) - h_ahead - (n_splits - i) * step
        train = ts_series.iloc[:train_end_idx]
        test = ts_series.iloc[train_end_idx:train_end_idx + h_ahead]
        
        if len(test) < h_ahead:
            continue
        
        y_pred = model_func(train, h_ahead).values.ravel()
        y_true = test.values
        
        residuals = y_true - y_pred
        all_residuals.append(residuals)
    
    all_residuals = np.concatenate(all_residuals)
    
    quantile_dict = {}
    for q in quantiles:
        quantile_dict[f'q{int(q*100)}'] = np.quantile(all_residuals, q)
    
    return {
        'residuals': all_residuals,
        'mean_residual': np.mean(all_residuals),
        'std_residual': np.std(all_residuals),
        'quantiles': quantile_dict
    }


def probabilistic_forecast_fleet(ts_series, target_name='Consumption', 
                                 split_date='2023-03-15 23:00:00', h_ahead=24):
    """General probability forecasts for the fleet"""
    
    print(f"\n{'='*55}")
    print(f"PROBABILISTIC FORECAST - {target_name.upper()}")
    print(f"{'='*55}\n")
    
    split_datetime = pd.to_datetime(split_date)
    train = ts_series.loc[:split_date]
    test = ts_series.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
    
    # Point forecast
    y_point = seasonal_naive(train, h_ahead).values.ravel()
    
    # Compute residual distribution
    residual_stats = compute_prediction_intervals(
        ts_series, seasonal_naive, split_date, h_ahead, n_splits=10
    )
    
    # Prediction intervals
    lower_5 = y_point + residual_stats['quantiles']['q5']
    lower_10 = y_point + residual_stats['quantiles']['q10']
    lower_25 = y_point + residual_stats['quantiles']['q25']
    median_pred = y_point + residual_stats['quantiles']['q50']
    upper_25 = y_point + residual_stats['quantiles']['q75']
    upper_10 = y_point + residual_stats['quantiles']['q90']
    upper_95 = y_point + residual_stats['quantiles']['q95']
    
    print(f"Residual Distribution Stats:")
    print(f"  Mean: {residual_stats['mean_residual']:.4f}")
    print(f"  Std:  {residual_stats['std_residual']:.4f}")
    print(f"\nPrediction Intervals (from residuals):")
    for q_name, q_val in residual_stats['quantiles'].items():
        print(f"  {q_name}: {q_val:.4f}")
    
    return {
        'point_forecast': y_point,
        'median_pred': median_pred,
        'lower_5': lower_5,
        'lower_10': lower_10,
        'lower_25': lower_25,
        'upper_25': upper_25,
        'upper_10': upper_10,
        'upper_95': upper_95,
        'actual': test.values,
        'residual_stats': residual_stats
    }


def plot_probabilistic_forecast(ts_series, target_name='Consumption', split_date='2023-03-15 23:00:00', h_ahead=24):
    """Plot fan chart con prediction intervals."""
    
    fc = probabilistic_forecast_fleet(ts_series, target_name, split_date, h_ahead)
    
    hours = np.arange(h_ahead)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Fan chart
    ax.fill_between(hours, fc['lower_5'], fc['upper_95'], alpha=0.1, 
                    color='blue', label='90% PI (5%-95%)')
    ax.fill_between(hours, fc['lower_10'], fc['upper_10'], alpha=0.15, 
                    color='blue', label='80% PI (10%-90%)')
    ax.fill_between(hours, fc['lower_25'], fc['upper_25'], alpha=0.2, 
                    color='blue', label='50% PI (25%-75%)')
    
    ax.plot(hours, fc['point_forecast'], color='blue', linewidth=2.5, 
            label='Point Forecast (S_Naive)', zorder=4)
    ax.plot(hours, fc['median_pred'], color='cyan', linewidth=1.5, 
            linestyle=':', label='Median Forecast', zorder=3)
    ax.plot(hours, fc['actual'], marker='o', color='black', linewidth=2.5, 
            markersize=6, label='Actual', zorder=5)
    
    ax.set_title(f'{target_name}: Probabilistic 24h Forecast (Fan Chart)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Hour Ahead', fontsize=12)
    ax.set_ylabel(f'{target_name} (units)', fontsize=12)
    ax.legend(loc='best', fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, h_ahead, 2))
    
    plt.tight_layout()
    return fig, fc


def print_probabilistic_summary(fc_result, h_ahead=24):
    """Print report of prob forecasting"""
    
    print(f"\n{'='*55}")
    print("PROBABILISTIC FORECAST - DETAILED RESULTS")
    print(f"{'='*55}\n")
    
    print(f"{'Hour':>6} {'Actual':>10} {'Point':>10} {'5%':>10} {'10%':>10} {'Median':>10} {'90%':>10} {'95%':>10}")
    print(f"{'-'*55}")
    
    for i in range(h_ahead):
        print(f"{i:6d} {fc_result['actual'][i]:10.2f} {fc_result['point_forecast'][i]:10.2f} "
              f"{fc_result['lower_5'][i]:10.2f} {fc_result['lower_10'][i]:10.2f} "
              f"{fc_result['median_pred'][i]:10.2f} {fc_result['upper_10'][i]:10.2f} "
              f"{fc_result['upper_95'][i]:10.2f}")
    
    print(f"\n{'='*55}")
    print("COVERAGE ANALYSIS")
    print(f"{'='*55}\n")
    
    coverage_80 = np.sum((fc_result['actual'] >= fc_result['lower_10']) & 
                         (fc_result['actual'] <= fc_result['upper_10']))
    coverage_90 = np.sum((fc_result['actual'] >= fc_result['lower_5']) & 
                         (fc_result['actual'] <= fc_result['upper_95']))
    
    print(f"80% PI Coverage: {coverage_80}/{h_ahead} = {100*coverage_80/h_ahead:.1f}%")
    print(f"90% PI Coverage: {coverage_90}/{h_ahead} = {100*coverage_90/h_ahead:.1f}%")
    
    width_80 = np.mean(fc_result['upper_10'] - fc_result['lower_10'])
    width_90 = np.mean(fc_result['upper_95'] - fc_result['lower_5'])
    
    print(f"\n80% PI Avg Width: {width_80:.2f}")
    print(f"90% PI Avg Width: {width_90:.2f}")
