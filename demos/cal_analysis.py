import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from datetime import datetime, timedelta
from scipy import stats
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from general.utils import prt, get_data_path
from projects.data_manage.getting.get_fwd import get_fwd_prices

def calculate_hurst_exponent(series):
    """Calculate Hurst exponent using DETRENDED R/S method (bounds to [0,1])."""
    series = series.dropna()
    if len(series) < 50:
        return np.nan

    x = np.arange(len(series))
    slope, intercept = np.polyfit(x, series, 1)
    detrended = series - (slope * x + intercept)

    rs = []
    ta = []
    for i in range(50, len(detrended) + 1):
        sub = detrended[:i]
        mean = np.mean(sub)
        cum_dev = (sub - mean).cumsum()
        R = max(cum_dev) - min(cum_dev)
        S = np.std(sub, ddof=1)
        if S > 1e-10:
            rs.append(R / S)
            ta.append(i)

    if len(rs) < 10:
        return np.nan

    log_rs = np.log(rs)
    log_ta = np.log(ta)
    slope, _, _, _, _ = stats.linregress(log_ta, log_rs)

    return np.clip(slope, 0.0, 1.0)

def calculate_rolling_adf(series, window=90):
    """Rolling ADF test p-values (low = stationary)."""
    results = []
    for i in range(len(series)):
        if i < window:
            results.append(np.nan)
            continue
        window_data = series.iloc[i-window:i].dropna()
        if len(window_data) < 10:
            results.append(np.nan)
            continue
        try:
            _, pvalue, _, _, _, _ = adfuller(window_data)
            results.append(pvalue)
        except:
            results.append(np.nan)
    return pd.Series(results, index=series.index)

def calculate_rolling_feature(series, func, window=90, min_periods=30):
    """Generic rolling feature calculator."""
    results = []
    for i in range(len(series)):
        if i < window:
            results.append(np.nan)
            continue
        window_data = series.iloc[i-window:i]
        if len(window_data.dropna()) < min_periods:
            results.append(np.nan)
            continue
        try:
            results.append(func(window_data))
        except:
            results.append(np.nan)
    return pd.Series(results, index=series.index)

def calculate_half_life(series):
    """Estimate half-life of mean reversion using OLS (for AR(1) process)."""
    series = series.dropna()
    if len(series) < 10:
        return np.nan
    lagged = series.shift(1).dropna()
    delta = series.dropna() - series.shift(1).dropna()
    if len(lagged) < 2 or len(delta) < 2:
        return np.nan
    X = sm.add_constant(lagged)
    model = sm.OLS(delta, X).fit()
    gamma = model.params[1]
    if gamma >= 0:
        return np.nan
    return -np.log(2) / gamma

def calculate_meta_features(series, label):
    """Calculate comprehensive time-series meta features."""
    series = series.dropna()
    if len(series) == 0:
        return {f"empty_{label}": True}

    features = {
        f"count_{label}": len(series),
        f"mean_{label}": series.mean(),
        f"std_{label}": series.std(),
        f"min_{label}": series.min(),
        f"max_{label}": series.max(),
        f"median_{label}": series.median(),
        f"q25_{label}": series.quantile(0.25),
        f"q75_{label}": series.quantile(0.75),
        f"skew_{label}": series.skew(),
        f"kurtosis_{label}": series.kurtosis(),
        f"range_{label}": series.max() - series.min(),
        f"iqr_{label}": series.quantile(0.75) - series.quantile(0.25),
        f"cv_{label}": (series.std() / abs(series.mean())) * 100 if series.mean() != 0 else np.nan,
    }

    try:
        features[f"hurst_{label}"] = calculate_hurst_exponent(series)
    except:
        features[f"hurst_{label}"] = np.nan

    try:
        features[f"autocorr_lag1_{label}"] = series.autocorr(1)
    except:
        features[f"autocorr_lag1_{label}"] = np.nan

    try:
        features[f"autocorr_lag5_{label}"] = series.autocorr(5)
    except:
        features[f"autocorr_lag5_{label}"] = np.nan

    try:
        features[f"half_life_{label}"] = calculate_half_life(series)
    except:
        features[f"half_life_{label}"] = np.nan

    if len(series) >= 90:
        rolling_std = series.rolling(90).std().dropna()
        features[f"rolling_std_90d_{label}"] = rolling_std.mean()
        features[f"rolling_std_90d_max_{label}"] = rolling_std.max()
    else:
        features[f"rolling_std_90d_{label}"] = np.nan
        features[f"rolling_std_90d_max_{label}"] = np.nan

    return features

def filter_by_horizon(df, horizon_days, quote_col="quotation_date"):
    """Filter DataFrame by time horizon in days."""
    if len(df) == 0:
        return df
    latest = pd.to_datetime(df[quote_col]).max()
    cutoff = latest - timedelta(days=horizon_days)
    df = df.copy()
    df[quote_col] = pd.to_datetime(df[quote_col])
    return df[df[quote_col] >= cutoff]

def plot_analysis(df, cc1, cc2, years, ptype):

    title=f"{cc1}-{cc2} {ptype} Spread ({years[0]}-{years[-1]})"
    """
    Clean, production-grade visualization for spread analysis.
    Displays the core Spread Asset alongside Bollinger Bands and MAs on the left,
    and statistical attributes tracking on the right.
    """
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    c_main = "#1e293b"
    c_ma30 = "#10b981"
    c_ma50 = "#0284c7"
    c_ma200 = "#dc2626"
    c_hurst = "#7c3aed"
    c_vol = "#db2777"
    c_adf = "#ea580c"
    c_grid = "#e2e8f0"
    
    df = df.copy()
    df["quotation_date"] = pd.to_datetime(df["quotation_date"])
    df = df.sort_values("quotation_date").set_index("quotation_date")
    spread = df["spread"]

    rolling_hurst = calculate_rolling_feature(spread, calculate_hurst_exponent, window=90, min_periods=50)
    rolling_vol = spread.rolling(30).std()
    rolling_adf = calculate_rolling_adf(spread, window=90)

    bb_mid = spread.rolling(20).mean()
    bb_std = spread.rolling(20).std()
    bb_up, bb_low = bb_mid + 2 * bb_std, bb_mid - 2 * bb_std

    ma30 = spread.rolling(30).mean()
    ma50 = spread.rolling(50).mean()
    ma200 = spread.rolling(200).mean()

    fig, axs = plt.subplots(3, 2, figsize=(22, 12), sharex=True, 
                            gridspec_kw={'width_ratios': [1.3, 1], 'hspace': 0.18, 'wspace': 0.18})
    
    gs = axs[0, 0].get_gridspec()
    for ax in axs[:, 0]:
        ax.remove()
    ax_price = fig.add_subplot(gs[:, 0])
    
    ax_hurst, ax_vol, ax_adf = axs[0, 1], axs[1, 1], axs[2, 1]

    ax_price.plot(df.index, spread, color=c_main, linewidth=1.8, label="Spread")
    ax_price.plot(df.index, bb_mid, color="#64748b", linestyle=":", linewidth=1.2, label="BB Mid (20)")
    ax_price.fill_between(df.index, bb_up, bb_low, alpha=0.06, color=c_main, label="BB Range (2σ)")
    
    ax_price.plot(df.index, ma30, color=c_ma30, linestyle="-", linewidth=1.2, label="MA 30")
    ax_price.plot(df.index, ma50, color=c_ma50, linestyle="-", linewidth=1.2, label="MA 50")
    ax_price.plot(df.index, ma200, color=c_ma200, linestyle="--", linewidth=1.5, label="MA 200")
    
    ax_price.set_ylabel("Spread (€/MWh)", fontweight="bold", fontsize=12)
    ax_price.set_title(title, fontsize=18, fontweight="bold", loc="left", pad=15)
    ax_price.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", fontsize=10)
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    ax_hurst.plot(df.index, rolling_hurst, color=c_hurst, linewidth=1.5)
    ax_hurst.axhline(0.5, color="#64748b", linestyle=":", linewidth=1)
    ax_hurst.fill_between(df.index, rolling_hurst, 0.5, where=(rolling_hurst < 0.5), color="#10b981", alpha=0.05)
    ax_hurst.fill_between(df.index, rolling_hurst, 0.5, where=(rolling_hurst > 0.5), color="#ef4444", alpha=0.05)
    ax_hurst.set_ylabel("Hurst Exponent", fontweight="bold")
    ax_hurst.set_title("Statistical Properties (90-Day Rolling Windows)", fontsize=13, fontweight="bold", loc="left", pad=12)
    ax_hurst.set_ylim(0, 1)

    ax_vol.plot(df.index, rolling_vol, color=c_vol, linewidth=1.5)
    ax_vol.fill_between(df.index, rolling_vol, alpha=0.08, color=c_vol)
    ax_vol.set_ylabel("30d Volatility (€/MWh)", fontweight="bold")

    ax_adf.plot(df.index, rolling_adf, color=c_adf, linewidth=1.5)
    ax_adf.axhline(0.05, color="#dc2626", linestyle="--", linewidth=1, label="α = 0.05")
    ax_adf.fill_between(df.index, rolling_adf, 0.05, where=(rolling_adf <= 0.05), color="#10b981", alpha=0.1)
    ax_adf.set_ylabel("ADF p-value", fontweight="bold")
    ax_adf.set_ylim(-0.02, 1.02)
    ax_adf.legend(loc="upper right", facecolor="white", edgecolor="none")

    all_active_axes = [ax_price, ax_hurst, ax_vol, ax_adf]
    for ax in all_active_axes:
        ax.grid(True, color=c_grid, linestyle="-", linewidth=0.5)
        ax.set_facecolor("#ffffff")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color(c_grid)
        ax.spines['bottom'].set_color(c_grid)
        
        if ax in [ax_price, ax_adf]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            ax.xaxis.set_major_locator(mticker.MaxNLocator(6))
            
    fig.patch.set_facecolor("#f8fafc") 
    
    plt.savefig(f"{cc1}_{cc2}_spread_analysis_clean.png", dpi=300, bbox_inches="tight")
    plt.show()

def main(cc1="FR", cc2="DE", ptype="CAL"):
    df_fwd = get_fwd_prices(prod_type=ptype)

    years = [2024, 2025, 2026, 2027]

    years_str = [str(y) for y in years]

    df_fwd_2 = df_fwd[
        df_fwd["country"].str.contains(cc2, na=False) &
        df_fwd["product_date"].astype(str).str[:4].isin(years_str)  # Fixed: Added .str before [:4]
    ]

    df_fwd_1 = df_fwd[
        df_fwd["country"].str.contains(cc1, na=False) &
        df_fwd["product_date"].astype(str).str[:4].isin(years_str)  # Fixed: Added .str before [:4]
    ]

    print(df_fwd_2.head())
    print(df_fwd_1.head())

    df_fwd_2_1 = pd.merge(
        df_fwd_2[["quotation_date", "price", "product_date"]],
        df_fwd_1[["quotation_date", "price", "product_date"]],
        on=["quotation_date", "product_date"],
        suffixes=("_2", "_1")
    )
    df_fwd_2_1["spread"] = df_fwd_2_1["price_2"] - df_fwd_2_1["price_1"]

    df_unique = df_fwd_2_1.sort_values("quotation_date").drop_duplicates(
        subset="quotation_date", keep="last"
    ).reset_index(drop=True)

    plot_analysis(df_unique, cc1, cc2, years=years, ptype=ptype)

    horizons = {
        "last_month": 30, "last_3_months": 90, "last_6_months": 180,
        "last_year": 365, "last_2_years": 730, "all_time": None,
    }

    print("\n" + "=" * 70)
    print(f"TIME-SERIES META FEATURES FOR {cc1}-{cc2} SPREAD (2024-2027)")
    print("=" * 70)

    for label, days in horizons.items():
        filtered = filter_by_horizon(df_unique, days) if days else df_unique
        if len(filtered) > 0:
            features = calculate_meta_features(filtered["spread"], label)
            print(f"\n📊 {label.replace('_', ' ').title()}:")
            for k, v in features.items():
                print(f"   {k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"   {k}: {v}")
        else:
            print(f"\n{label.replace('_', ' ').title()}: Insufficient data")

if __name__ == "__main__":
    #main("IT", "HU", "CAL")
    main("FR", "DE", "CAL")
    #main("IT", "HU", "M")
    #main("FR", "DE", "M")

