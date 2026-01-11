import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

# Handle Chronos import gracefully for environments where it isn't installed
try:
    from chronos import BaseChronosPipeline
except ImportError:
    BaseChronosPipeline = None
    print("Warning: 'chronos' library not found. Chronos-related functions will not work.")

# ============================================================================
# BASIC UTILS & BENCHMARK MODELS
# ============================================================================

def cast_df(y, df):
    """Helper to format forecast arrays into timestamps DataFrames."""
    h = len(y)
    return pd.DataFrame(y, index=pd.date_range(
        start=df.index[-1] + pd.Timedelta(hours=1), periods=h, freq='h'))

def naive(x, h): 
    """Naive Forecast: Uses the last observed value."""
    return cast_df(np.repeat(x.iloc[-1], h), x)

def mean_fc(x, h): 
    """Mean Forecast: Uses the historical mean."""
    return cast_df(np.repeat(x.mean(), h), x)

def seasonal_naive(x, h, m=24):
    """
    Seasonal Naive Forecast.
    Repeats the last 'm' periods (default 24h for daily seasonality).
    """
    values = x.iloc[-m:].values
    y_hat = np.tile(values, int(np.ceil(h/m)))[:h]
    return cast_df(y_hat, x)

def calculate_nmae(y_true, y_pred, y_train_mean):
    """
    Calculates Normalized Mean Absolute Error (nMAE).
    Normalized by the MAE of a naive baseline (mean forecast).
    """
    mae = np.mean(np.abs(y_true - y_pred))
    mae_baseline = np.mean(np.abs(y_true - np.mean(y_true)))  
    return mae / mae_baseline if mae_baseline != 0 else np.nan

def rolling_window_cv(ts_series, model_func, h_ahead=24, n_splits=10):
    """
    Time Series Cross-Validation with an expanding window.
    Splits data into 'n_splits' folds to evaluate model stability.
    """
    results = {'mae': [], 'nmae': [], 'actuals': [], 'forecasts': []}
    
    # Calculate step size to ensure we use the whole dataset
    step = (len(ts_series) - h_ahead) // (n_splits + 1)
    
    for i in range(n_splits):
        train_end_idx = len(ts_series) - h_ahead - (n_splits - i) * step
        train = ts_series.iloc[:train_end_idx]
        test = ts_series.iloc[train_end_idx:train_end_idx + h_ahead]
        
        # Skip incomplete folds
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

# ============================================================================
# BENCHMARKING SUITES
# ============================================================================

def benchmark_per_rail(ts_data, target_name, split_date='2023-03-15 23:00:00', h_ahead=24):
    """
    Iterates through each rail (column) and computes benchmark metrics.
    """
    print(f"BENCHMARK NAIVE - {target_name.upper()}")
    
    bench_results = {}
    split_datetime = pd.to_datetime(split_date)
    
    for rail in ts_data.columns:
        ts = ts_data[rail]
        train = ts.loc[:split_date]
        test = ts.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
        
        if len(test) < h_ahead:
            continue
        
        y_true = test.values
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
    Benchmark for aggregated fleet level using Rolling Window CV + single Test set.
    """
    print(f"BENCHMARK FLEET - {target_name.upper()}")
    
    split_datetime = pd.to_datetime(split_date)
    
    # 1. Cross-validation
    cv_naive = rolling_window_cv(ts_series, naive, h_ahead=h_ahead, n_splits=10)
    cv_mean = rolling_window_cv(ts_series, mean_fc, h_ahead=h_ahead, n_splits=10)
    cv_snai = rolling_window_cv(ts_series, seasonal_naive, h_ahead=h_ahead, n_splits=10)
    
    print("CV (10-fold) nMAE:")
    print(f"  Naive:   {np.nanmean(cv_naive['nmae']):.4f} ± {np.nanstd(cv_naive['nmae']):.4f}")
    print(f"  Mean:    {np.nanmean(cv_mean['nmae']):.4f} ± {np.nanstd(cv_mean['nmae']):.4f}")
    print(f"  S_Naive: {np.nanmean(cv_snai['nmae']):.4f} ± {np.nanstd(cv_snai['nmae']):.4f}")
    
    # 2. Single Test set evaluation
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
    """Visualizes the benchmark forecasts against actual data."""
    
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

def compute_prediction_intervals(ts_series, model_func, split_date='2023-03-15 23:00:00', 
                                 h_ahead=24, n_splits=10, quantiles=[0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]):
    """
    Computes prediction intervals based on historical residuals from Cross-Validation.
    """
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
        all_residuals.append(y_true - y_pred)
    
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


def probabilistic_forecast_fleet(ts_series, target_name='Consumption', split_date='2023-03-15 23:00:00', h_ahead=24):
    """
    Generates a probabilistic forecast for the fleet.
    """
    print(f"PROBABILISTIC - {target_name.upper()}")
    
    split_datetime = pd.to_datetime(split_date)
    train = ts_series.loc[:split_date]
    test = ts_series.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
    
    # Point forecast
    y_point = seasonal_naive(train, h_ahead).values.ravel()
    
    # Compute residual distribution
    residual_stats = compute_prediction_intervals(
        ts_series, seasonal_naive, split_date, h_ahead, n_splits=10
    )
    
    # Add residuals to point forecast
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
    """Generates a Fan Chart visualization with prediction intervals."""
    
    fc = probabilistic_forecast_fleet(ts_series, target_name, split_date, h_ahead)
    
    hours = np.arange(h_ahead)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Fan chart shading
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
    """Prints a concise summary and coverage metrics."""
    
    print("PROBABILISTIC FORECAST - SUMMARY")
    
    # Create a DataFrame for cleaner display of first few rows
    df_summary = pd.DataFrame({
        'Actual': fc_result['actual'],
        'Point': fc_result['point_forecast'],
        'Lo_10': fc_result['lower_10'],
        'Hi_10': fc_result['upper_10']
    })
    
    print(f"First 5 hours:\n{df_summary.head().round(2)}")
    
    # Coverage
    cov_80 = np.sum((fc_result['actual'] >= fc_result['lower_10']) & (fc_result['actual'] <= fc_result['upper_10']))
    cov_90 = np.sum((fc_result['actual'] >= fc_result['lower_5']) & (fc_result['actual'] <= fc_result['upper_95']))
    
    width_80 = np.mean(fc_result['upper_10'] - fc_result['lower_10'])
    
    print(f"Coverage | 80% PI: {100*cov_80/h_ahead:.1f}% | 90% PI: {100*cov_90/h_ahead:.1f}%")
    print(f"Avg Width (80% PI): {width_80:.2f}")


# ============================================================================
# ML SECTION - CHRONOS 2
# ============================================================================

def train_chronos2_fleet(fleet_ts, target_name='consumption', split_date='2023-03-15 23:00:00', h_ahead=24):
    """
    Uses the Chronos 2 foundation model for fleet-level forecasting.
    """
    
    if BaseChronosPipeline is None:
        return None

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
        # Prepare context
        df_context = pd.DataFrame({
            'Date Time': train.index,
            'item_id': 'ev_fleet',
            target_name: train.values
        }).reset_index(drop=True)
        
        # Inference
        cron_pred_df = pipeline.predict_df(
            df_context,
            prediction_length=h_ahead,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            id_column="item_id",
            timestamp_column="Date Time",
            target=target_name,
        )
        
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
    """Runs Chronos 2 forecast for each rail individually."""
    
    if BaseChronosPipeline is None:
        return None, None

    print(f"CHRONOS 2 PER RAIL - {target_name.upper()}")
    split_datetime = pd.to_datetime(split_date)
    results = {}
    
    try:
        pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
    except Exception:
        return None, None
    
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
    """Probabilistic forecast using Chronos 2 native quantiles."""
    
    if BaseChronosPipeline is None:
        return None

    print(f"PROB FORECAST (CHRONOS) - {target_name.upper()}")
    split_datetime = pd.to_datetime(split_date)
    train = fleet_ts.loc[:split_date]
    test = fleet_ts.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:h_ahead]
    
    try:
        pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        df_context = pd.DataFrame({
            'Date Time': train.index, 'item_id': 'ev_fleet', target_name.lower(): train.values
        }).reset_index(drop=True)
        
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
    """Plot fan chart using Chronos 2 generated quantiles."""
    
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
    """Print concise results for Chronos 2."""
    if fc_result is None: return
    
    print("CHRONOS PROB FORECAST - SUMMARY")
    df_summary = pd.DataFrame({
        'Actual': fc_result['actual'], 'Point': fc_result['point_forecast'],
        'Lo_10': fc_result['lower_10'], 'Hi_10': fc_result['upper_10']
    })
    print(f"First 5 hours:\n{df_summary.head().round(2)}")
    
    cov_80 = np.sum((fc_result['actual'] >= fc_result['lower_10']) & (fc_result['actual'] <= fc_result['upper_10']))
    print(f"Coverage 80% PI: {100*cov_80/h_ahead:.1f}%")


def compare_all_models_w_chronos2(fleet_ts, target_name='Consumption', split_date='2023-03-15 23:00:00'):
    """Compare S_Naive, ARIMA, and Chronos 2."""
    
    print(f"MODEL COMPARISON - {target_name.upper()}")
    
    split_datetime = pd.to_datetime(split_date)
    train = fleet_ts.loc[:split_date]
    test = fleet_ts.loc[split_datetime + pd.Timedelta(hours=1):].iloc[:24]
    
    results = {}
    
    # S_Naive
    y_snai = seasonal_naive(train, 24).values.ravel()
    results['S_Naive'] = {'nMAE': calculate_nmae(test.values, y_snai, train.mean())}
    print(f"  S_Naive:  nMAE={results['S_Naive']['nMAE']:.4f}")
    
    # ARIMA
    try:
        from statsmodels.tsa.arima.model import ARIMA
        model = ARIMA(train, order=(2, 1, 2)).fit()
        y_arima = model.forecast(steps=24).values
        results['ARIMA'] = {'nMAE': calculate_nmae(test.values, y_arima, train.mean())}
        print(f"  ARIMA:    nMAE={results['ARIMA']['nMAE']:.4f}")
    except:
        print("  ARIMA:    Failed")
    
    # Chronos
    if BaseChronosPipeline:
        c_res = train_chronos2_fleet(fleet_ts, target_name.lower(), split_date)
        if c_res:
            results['Chronos2'] = {'nMAE': c_res['nmae']}
    
    if results:
        df_res = pd.DataFrame(results).T.sort_values('nMAE')
        best = df_res.index[0]
        imp = (results['S_Naive']['nMAE'] - df_res.iloc[0,0]) / results['S_Naive']['nMAE'] * 100
        print(f"BEST: {best} (Imp vs S_Naive: {imp:+.1f}%)")
        return results, df_res
    return {}, pd.DataFrame()


def predict_future_chronos2(ts_data, target_name, h_ahead=24):
    """Predicts future values using Chronos 2."""
    if BaseChronosPipeline is None: return None

    try:
        pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-2", device_map="cpu")
        print(f"FUTURE FORECAST ({h_ahead}h) - {target_name.upper()}")
        
        context_df = pd.DataFrame({
            'Date Time': pd.to_datetime(ts_data.index), 'item_id': 'ev_fleet', target_name: ts_data.values
        }).reset_index(drop=True)
        
        future_predictions_df = pipeline.predict_df(
            context_df, prediction_length=h_ahead, quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9],
            id_column="item_id", timestamp_column="Date Time", target=target_name
        )
        
        future_predictions_df.index = pd.date_range(ts_data.index[-1] + pd.Timedelta(hours=1), periods=h_ahead, freq='h')
        print(future_predictions_df[['0.5']].rename(columns={'0.5': 'Median'}).head())
        return future_predictions_df

    except Exception as e:
        print(f"Error: {str(e)[:100]}")
        return None

def plot_future_forecast(ts_data, forecast_df, target_name):
    """Plots historical data vs future predictions."""
    plt.figure(figsize=(15, 7))
    plt.plot(ts_data.index, ts_data.values, label='History', color='black', alpha=0.8)
    
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