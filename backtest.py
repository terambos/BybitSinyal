"""
Backtest - Kalman Trend + QQE Momentum + S/R Bolgesi Stratejisi
==================================================================
Canli botla (bybit_signal_bot.py) BIREBIR AYNI mantigi gecmis veriye
uygular. Her sinyal icin, sonraki 3 gun icinde HERHANGI BIR anda kac %
lehte hareket gorduğunu (MFE) ve yolda ne kadar aleyhe gidildigini (MAE)
olcer - "sinyali gorup kendi analizinle girip %5-10'da cikma" kullanim
seklini yansitir.

ONEMLI KISITLAMA: S/R bolgeleri, her coin icin TUM 4 saatlik gecmisten
(730 gun) hesaplanir - yani gelecekteki fiyat hareketleri de bolge
tespitine katkida bulunmus olabilir (hafif ileriye-bakis / lookahead).
Canli botta bu sorun yok (o an sadece o ana kadarki veriyi gorur).
Bu yuzden backtest sonuclari gercek performanstan biraz iyimser olabilir,
yorumlarken bunu akilda tut.

Sonuc: results/backtest_sonuclari.csv ve konsola ozet istatistikler.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# AYARLAR (bybit_signal_bot.py ile birebir ayni)
# ----------------------------------------------------------------------

CRYPTO_SYMBOLS = [
    "BTC", "ETH", "XRP", "SUI", "HYPE", "AVAX", "ARB", "APT",
    "FET", "RENDER", "TIA", "DOGE", "SOL",
    "LINK", "DOT", "NEAR", "INJ", "OP", "SEI",
    "BNB", "SHIB", "TAO", "ENA", "ATOM", "ALGO",
]

YF_SUFFIX = "-USD"
YF_TICKER_OVERRIDES = {
    "SUI": "SUI20947",
    "HYPE": "HYPE32196",
    "APT": "APT21794",
    "TAO": "TAO22974",
}
TREND_PERIOD = "730d"
TREND_INTERVAL = "1h"
TRIGGER_PERIOD = "60d"
TRIGGER_INTERVAL = "15m"

QQE_RSI_PERIOD = 14
QQE_SMOOTHING = 5
QQE_FACTOR = 4.236
QQE_WILDER_PERIOD = QQE_RSI_PERIOD * 2 - 1

KALMAN_PROCESS_VAR = 1e-5
KALMAN_MEASUREMENT_VAR = 1e-2
KALMAN_SLOPE_LOOKBACK = 5

SR_SWING_ORDER = 3
SR_CLUSTER_TOLERANCE = 0.015
SR_MIN_TOUCHES = 2
SR_STRONG_TOUCHES = 4
SR_PROXIMITY_PCT = 0.015

VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULT = 1.2

PUAN_KALMAN_1G = 3
PUAN_KALMAN_4S = 3
PUAN_HACIM = 2
PUAN_BOLGE_GUC_ZAYIF = 1
PUAN_BOLGE_GUC_GUCLU = 2
MAX_PUAN = PUAN_KALMAN_1G + PUAN_KALMAN_4S + PUAN_HACIM + PUAN_BOLGE_GUC_GUCLU
ESIK_ORTA_GUVEN = 5
ESIK_YUKSEK_GUVEN = 8

MFE_HORIZON = 288  # 3 gun (15dk mum sayisi)
HEDEF_YUZDELER = [3, 5, 7, 10]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ----------------------------------------------------------------------
# INDIKATORLER (canli botla ayni)
# ----------------------------------------------------------------------

def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).fillna(50)


def kalman_smooth(prices, process_var=KALMAN_PROCESS_VAR, measurement_var=KALMAN_MEASUREMENT_VAR):
    n = len(prices)
    xhat = np.zeros(n)
    P = np.zeros(n)
    xhat[0] = prices[0]
    P[0] = 1.0
    for k in range(1, n):
        P_minus = P[k - 1] + process_var
        K = P_minus / (P_minus + measurement_var)
        xhat[k] = xhat[k - 1] + K * (prices[k] - xhat[k - 1])
        P[k] = (1 - K) * P_minus
    return xhat


def kalman_trend_series(close: pd.Series, lookback=KALMAN_SLOPE_LOOKBACK):
    """Her nokta icin (gecmise bakarak, o ana kadarki veriyle) Kalman egimini dondurur."""
    xhat = kalman_smooth(close.values)
    xhat_s = pd.Series(xhat, index=close.index)
    egim = xhat_s - xhat_s.shift(lookback)
    return egim


def compute_qqe(close: pd.Series):
    rsi_val = rsi(close, QQE_RSI_PERIOD)
    rsi_ma = ema(rsi_val, QQE_SMOOTHING)
    atr_rsi = rsi_ma.diff().abs()
    ma_atr_rsi = atr_rsi.ewm(alpha=1 / QQE_WILDER_PERIOD, adjust=False).mean()
    dar = ma_atr_rsi.ewm(alpha=1 / QQE_WILDER_PERIOD, adjust=False).mean() * QQE_FACTOR

    rsi_ma_v = rsi_ma.values
    dar_v = dar.values
    n = len(close)
    longband = np.zeros(n)
    shortband = np.zeros(n)
    trend = np.zeros(n, dtype=int)

    for i in range(1, n):
        new_long = rsi_ma_v[i] - dar_v[i]
        new_short = rsi_ma_v[i] + dar_v[i]
        if rsi_ma_v[i - 1] > longband[i - 1] and rsi_ma_v[i] > longband[i - 1]:
            longband[i] = max(new_long, longband[i - 1])
        else:
            longband[i] = new_long
        if rsi_ma_v[i - 1] < shortband[i - 1] and rsi_ma_v[i] < shortband[i - 1]:
            shortband[i] = min(new_short, shortband[i - 1])
        else:
            shortband[i] = new_short
        if rsi_ma_v[i] > shortband[i - 1]:
            trend[i] = 1
        elif rsi_ma_v[i] < longband[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]

    trend_s = pd.Series(trend, index=close.index)
    bullish_cross = (trend_s == 1) & (trend_s.shift(1) != 1)
    bearish_cross = (trend_s == -1) & (trend_s.shift(1) != -1)
    return bullish_cross, bearish_cross


def find_sr_zones(df_htf):
    window = 2 * SR_SWING_ORDER + 1
    roll_max = df_htf["High"].rolling(window, center=True).max()
    roll_min = df_htf["Low"].rolling(window, center=True).min()
    swing_high_mask = df_htf["High"] == roll_max
    swing_low_mask = df_htf["Low"] == roll_min
    highs = sorted(df_htf.loc[swing_high_mask, "High"].dropna().tolist())
    lows = sorted(df_htf.loc[swing_low_mask, "Low"].dropna().tolist())

    def cluster(prices):
        zones = []
        group = []
        for p in prices:
            if not group or (p - group[-1]) / group[-1] <= SR_CLUSTER_TOLERANCE:
                group.append(p)
            else:
                if len(group) >= SR_MIN_TOUCHES:
                    zones.append((float(np.mean(group)), len(group)))
                group = [p]
        if len(group) >= SR_MIN_TOUCHES:
            zones.append((float(np.mean(group)), len(group)))
        return zones

    return cluster(lows), cluster(highs)


def zone_below(price, zones, max_pct=SR_PROXIMITY_PCT):
    cands = [(price - zp, zp, t) for zp, t in zones if zp <= price]
    if not cands:
        return None
    cands.sort()
    dist, zp, t = cands[0]
    return (zp, t) if dist / price <= max_pct else None


def zone_above(price, zones, max_pct=SR_PROXIMITY_PCT):
    cands = [(zp - price, zp, t) for zp, t in zones if zp >= price]
    if not cands:
        return None
    cands.sort()
    dist, zp, t = cands[0]
    return (zp, t) if dist / price <= max_pct else None


# ----------------------------------------------------------------------
# VERI CEKME
# ----------------------------------------------------------------------

def fetch_history(base, period, interval, retries=4):
    ticker = f"{YF_TICKER_OVERRIDES.get(base, base)}{YF_SUFFIX}"
    df = None
    for attempt in range(retries):
        try:
            df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        except Exception as e:
            print(f"  UYARI: {ticker} ({interval}) deneme {attempt+1} hata: {e}", file=sys.stderr)
            df = None
        if df is not None and not df.empty:
            break
        if attempt < retries - 1:
            wait = 5 * (attempt + 1)
            print(f"  {ticker} ({interval}) bos dondu, {wait}sn bekleyip tekrar denenecek...")
            time.sleep(wait)

    if df is None or df.empty:
        return None
    df = df.reset_index()
    dt_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={dt_col: "dt"})
    df["dt"] = pd.to_datetime(df["dt"], utc=True)
    return df[["dt", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)


def resample(df_1h, rule):
    df = df_1h.set_index("dt")
    out = df.resample(rule, label="right", closed="right").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()
    return out.reset_index()


# ----------------------------------------------------------------------
# TUM GECMIS ICIN TARAMA
# ----------------------------------------------------------------------

def backtest_symbol(base: str):
    df_trend_1h = fetch_history(base, TREND_PERIOD, TREND_INTERVAL)
    time.sleep(2)
    df_trig = fetch_history(base, TRIGGER_PERIOD, TRIGGER_INTERVAL)

    if df_trend_1h is None or df_trig is None:
        return pd.DataFrame(), False
    if len(df_trend_1h) < 200 * 24 or len(df_trig) < 300:
        return pd.DataFrame(), False

    df_4h = resample(df_trend_1h, "4h")
    df_1d = resample(df_trend_1h, "1D")
    if len(df_1d) < 30 or len(df_4h) < 60:
        return pd.DataFrame(), False

    # Kalman egimleri - HER NOKTA icin (o ana kadarki veriyle) hesaplanir
    egim_1d_series = kalman_trend_series(df_1d["Close"])
    egim_4h_series = kalman_trend_series(df_4h["Close"])
    df_1d = df_1d.copy()
    df_4h = df_4h.copy()
    df_1d["kalman_up"] = egim_1d_series > 0
    df_1d["kalman_down"] = egim_1d_series < 0
    df_4h["kalman_up"] = egim_4h_series > 0
    df_4h["kalman_down"] = egim_4h_series < 0

    # S/R bolgeleri - NOT: tum 4S gecmisinden (bkz. dosya basindaki kisitlama notu)
    destek_bolgeleri, direnc_bolgeleri = find_sr_zones(df_4h)

    trig = df_trig.sort_values("dt").reset_index(drop=True)
    trig = pd.merge_asof(trig, df_1d[["dt", "kalman_up", "kalman_down"]].rename(
        columns={"kalman_up": "kalman_up_1d", "kalman_down": "kalman_down_1d"}), on="dt", direction="backward")
    trig = pd.merge_asof(trig, df_4h[["dt", "kalman_up", "kalman_down"]].rename(
        columns={"kalman_up": "kalman_up_4h", "kalman_down": "kalman_down_4h"}), on="dt", direction="backward")
    trig["kalman_up_1d"] = trig["kalman_up_1d"].fillna(False)
    trig["kalman_down_1d"] = trig["kalman_down_1d"].fillna(False)
    trig["kalman_up_4h"] = trig["kalman_up_4h"].fillna(False)
    trig["kalman_down_4h"] = trig["kalman_down_4h"].fillna(False)

    qqe_bull, qqe_bear = compute_qqe(trig["Close"])
    vol_ma = trig["Volume"].rolling(VOLUME_MA_PERIOD).mean()
    volume_spike = (vol_ma > 0) & (trig["Volume"] >= VOLUME_SPIKE_MULT * vol_ma)

    closes = trig["Close"].values
    n = len(trig)

    kayitlar = []
    for i in range(n):
        if not (bool(qqe_bull.iloc[i]) or bool(qqe_bear.iloc[i])):
            continue
        price_now = closes[i]

        long_kapi = bool(qqe_bull.iloc[i]) and (zone_below(price_now, destek_bolgeleri) is not None)
        short_kapi = bool(qqe_bear.iloc[i]) and (zone_above(price_now, direnc_bolgeleri) is not None)
        if not (long_kapi or short_kapi):
            continue

        long_puan = short_puan = 0
        long_bolge = short_bolge = None

        if long_kapi:
            long_bolge = zone_below(price_now, destek_bolgeleri)
            if trig["kalman_up_1d"].iloc[i]:
                long_puan += PUAN_KALMAN_1G
            if trig["kalman_up_4h"].iloc[i]:
                long_puan += PUAN_KALMAN_4S
            if volume_spike.iloc[i]:
                long_puan += PUAN_HACIM
            long_puan += PUAN_BOLGE_GUC_GUCLU if long_bolge[1] >= SR_STRONG_TOUCHES else PUAN_BOLGE_GUC_ZAYIF

        if short_kapi:
            short_bolge = zone_above(price_now, direnc_bolgeleri)
            if trig["kalman_down_1d"].iloc[i]:
                short_puan += PUAN_KALMAN_1G
            if trig["kalman_down_4h"].iloc[i]:
                short_puan += PUAN_KALMAN_4S
            if volume_spike.iloc[i]:
                short_puan += PUAN_HACIM
            short_puan += PUAN_BOLGE_GUC_GUCLU if short_bolge[1] >= SR_STRONG_TOUCHES else PUAN_BOLGE_GUC_ZAYIF

        if long_puan >= short_puan and long_kapi:
            direction, puan = "LONG", long_puan
        elif short_kapi:
            direction, puan = "SHORT", short_puan
        else:
            continue

        if puan < ESIK_ORTA_GUVEN:
            continue

        guven = "YUKSEK" if puan >= ESIK_YUKSEK_GUVEN else "ORTA"

        # MFE/MAE
        start, end = i + 1, min(i + 1 + MFE_HORIZON, n)
        if start >= end:
            continue
        window = closes[start:end]
        if direction == "LONG":
            pct_path = (window / price_now - 1) * 100
        else:
            pct_path = (price_now - window) / price_now * 100
        mfe = float(np.max(pct_path))
        mae = float(np.min(pct_path))

        kayit = {
            "symbol": base, "dt": trig["dt"].iloc[i], "direction": direction,
            "puan": puan, "guven": guven, "Close": price_now,
            "mfe_%": round(mfe, 2), "mae_%": round(mae, 2),
        }
        for h in HEDEF_YUZDELER:
            kayit[f"hedef_%{h}_ulasti"] = mfe >= h
        kayitlar.append(kayit)

    if not kayitlar:
        return pd.DataFrame(), True
    return pd.DataFrame(kayitlar), True


# ----------------------------------------------------------------------
# ANA AKIS
# ----------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_signals = []
    basarili_coinler, veri_alinamayan = [], []

    for idx, base in enumerate(CRYPTO_SYMBOLS):
        print(f"\n[{idx+1}/{len(CRYPTO_SYMBOLS)}] Taraniyor: {base}...")
        try:
            res, veri_tamam = backtest_symbol(base)
        except Exception as e:
            print(f"HATA ({base}): {e}", file=sys.stderr)
            veri_alinamayan.append(base)
            continue

        if not veri_tamam:
            print(f"  -> VERI ALINAMADI")
            veri_alinamayan.append(base)
        else:
            basarili_coinler.append(base)
            if res is not None and not res.empty:
                all_signals.append(res)
                print(f"  -> {len(res)} sinyal bulundu")
            else:
                print(f"  -> veri tamam, ama hic sinyal olusmamis")

        if idx < len(CRYPTO_SYMBOLS) - 1:
            time.sleep(3)

    print("\n" + "=" * 60)
    print(f"VERI DURUMU: {len(basarili_coinler)}/{len(CRYPTO_SYMBOLS)} coin basariyla tarandi")
    if veri_alinamayan:
        print(f"Veri alinamayan coinler: {', '.join(veri_alinamayan)}")
    print("=" * 60)

    if not all_signals:
        print("\nHICBIR SINYAL BULUNAMADI.")
        return

    df = pd.concat(all_signals, ignore_index=True)
    df = df.sort_values("dt")
    out_path = os.path.join(OUT_DIR, "backtest_sonuclari.csv")
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"TOPLAM SINYAL SAYISI: {len(df)}")
    print(f"LONG: {(df['direction']=='LONG').sum()}  |  SHORT: {(df['direction']=='SHORT').sum()}")
    print(f"YUKSEK GUVEN: {(df['guven']=='YUKSEK').sum()}  |  ORTA GUVEN: {(df['guven']=='ORTA').sum()}")
    print("=" * 60)

    print("\nSembol basina sinyal sayisi:")
    print(df["symbol"].value_counts().to_string())

    print("\n" + "-" * 60)
    print(f"HEDEF YAKALAMA ORANI (sinyalden sonraki {MFE_HORIZON//96} gun icinde, HERHANGI BIR anda):")
    print("-" * 60)
    for h in HEDEF_YUZDELER:
        col = f"hedef_%{h}_ulasti"
        oran = df[col].mean() * 100
        print(f"  %{h}'e ulasan sinyal orani: %{oran:.1f}  ({df[col].sum()}/{len(df)} sinyal)")

    print(f"\nOrtalama MFE: %{df['mfe_%'].mean():.2f}  |  Medyan MFE: %{df['mfe_%'].median():.2f}")
    print(f"Ortalama MAE: %{df['mae_%'].mean():.2f}  |  Medyan MAE: %{df['mae_%'].median():.2f}")

    print("\n" + "-" * 60)
    print("GUVEN SEVIYESINE GORE HEDEF YAKALAMA:")
    print("-" * 60)
    for guven_lvl in ["YUKSEK", "ORTA"]:
        alt = df[df["guven"] == guven_lvl]
        if alt.empty:
            print(f"{guven_lvl}: sinyal yok")
            continue
        print(f"{guven_lvl}: {len(alt)} sinyal")
        for h in [5, 10]:
            col = f"hedef_%{h}_ulasti"
            print(f"    -> %{h} hedefe ulasan oran: %{alt[col].mean()*100:.1f}")

    print(f"\nTum detaylar kaydedildi: {out_path}")


if __name__ == "__main__":
    main()
