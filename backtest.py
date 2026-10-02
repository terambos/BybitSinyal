"""
Backtest v3 - Bolge Gucu + Yakinlik + Momentum + Hacim odakli strateji
========================================================================
Canli botla birebir ayni mantik. Trend puanlamadan cikarildi (onceki
testte anti-korelasyon gosterdi). Sonuc: results/backtest_sonuclari.csv
"""

import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

CRYPTO_SYMBOLS = [
    "BTC", "ETH", "XRP", "SUI", "HYPE", "AVAX", "ARB", "APT",
    "FET", "RENDER", "TIA", "DOGE", "SOL",
    "LINK", "DOT", "NEAR", "INJ", "OP", "SEI",
    "BNB", "SHIB", "TAO", "ENA", "ATOM", "ALGO",
]

YF_SUFFIX = "-USD"
YF_TICKER_OVERRIDES = {
    "SUI": "SUI20947", "HYPE": "HYPE32196", "APT": "APT21794", "TAO": "TAO22974",
}
TREND_PERIOD = "730d"
TREND_INTERVAL = "1h"
TRIGGER_PERIOD = "60d"
TRIGGER_INTERVAL = "15m"

QQE_RSI_PERIOD = 14
QQE_SMOOTHING = 5
QQE_FACTOR = 4.236
QQE_WILDER_PERIOD = QQE_RSI_PERIOD * 2 - 1

SR_SWING_ORDER_4H = 3
SR_SWING_ORDER_1D = 2
SR_DAILY_TOUCH_BONUS = 2
MIN_TP_DISTANCE_PCT = 0.03
SR_CLUSTER_TOLERANCE = 0.015
SR_MIN_TOUCHES = 3
SR_PROXIMITY_PCT = 0.010

VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULT = 1.5

MAX_PUAN = 10
ESIK_ORTA_GUVEN = 6
QQE_MIN_STRENGTH = 0.15
ATR_PERIOD = 14
ATR_LOOKBACK = 50
ATR_MIN_RATIO = 0.6
SYMBOL_COOLDOWN_MS = 4 * 60 * 60 * 1000
ESIK_YUKSEK_GUVEN = 8

MFE_HORIZON = 288
HEDEF_YUZDELER = [3, 5, 7, 10]

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


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


def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def compute_qqe(close):
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

    shortband_prev = pd.Series(shortband, index=close.index).shift(1)
    longband_prev = pd.Series(longband, index=close.index).shift(1)
    dar_safe = dar.replace(0, np.nan)
    bull_strength = ((rsi_ma - shortband_prev) / dar_safe).clip(lower=0)
    bear_strength = ((longband_prev - rsi_ma) / dar_safe).clip(lower=0)
    cross_strength = np.where(bullish_cross, bull_strength, np.where(bearish_cross, bear_strength, 0.0))
    cross_strength = pd.Series(cross_strength, index=close.index).fillna(0.0)

    return bullish_cross, bearish_cross, cross_strength, trend_s


def find_sr_zones(df_htf, swing_order):
    window = 2 * swing_order + 1
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
    return (zp, t, dist / price) if dist / price <= max_pct else None


def zone_above(price, zones, max_pct=SR_PROXIMITY_PCT):
    cands = [(zp - price, zp, t) for zp, t in zones if zp >= price]
    if not cands:
        return None
    cands.sort()
    dist, zp, t = cands[0]
    return (zp, t, dist / price) if dist / price <= max_pct else None


def nearest_above_any(price, zones):
    cands = [(zp - price, zp, t) for zp, t in zones if zp > price]
    if not cands:
        return None
    cands.sort()
    return cands[0][1], cands[0][2]


def nearest_below_any(price, zones):
    cands = [(price - zp, zp, t) for zp, t in zones if zp < price]
    if not cands:
        return None
    cands.sort()
    return cands[0][1], cands[0][2]


def puan_bolge_guc(touches):
    if touches >= 6:
        return 3
    if touches >= 4:
        return 2
    return 1


def puan_yakinlik(dist_pct):
    if dist_pct <= 0.005:
        return 3
    if dist_pct <= 0.010:
        return 2
    return 1


def puan_momentum(strength):
    if strength >= 0.5:
        return 2
    if strength >= 0.15:
        return 1
    return 0


def puan_hacim(spike):
    return 2 if spike else 0


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


def backtest_symbol(base):
    df_trend_1h = fetch_history(base, TREND_PERIOD, TREND_INTERVAL)
    time.sleep(2)
    df_trig = fetch_history(base, TRIGGER_PERIOD, TRIGGER_INTERVAL)

    if df_trend_1h is None or df_trig is None:
        return pd.DataFrame(), False
    if len(df_trend_1h) < 200 * 24 or len(df_trig) < 300:
        return pd.DataFrame(), False

    df_4h = resample(df_trend_1h, "4h")
    df_1d = resample(df_trend_1h, "1D")
    if len(df_4h) < 60:
        return pd.DataFrame(), False

    destek_4h, direnc_4h = find_sr_zones(df_4h, SR_SWING_ORDER_4H)
    destek_bolgeleri, direnc_bolgeleri = list(destek_4h), list(direnc_4h)
    if len(df_1d) >= 2 * SR_SWING_ORDER_1D + 5:
        destek_1d, direnc_1d = find_sr_zones(df_1d, SR_SWING_ORDER_1D)
        destek_bolgeleri += [(zp, t + SR_DAILY_TOUCH_BONUS) for zp, t in destek_1d]
        direnc_bolgeleri += [(zp, t + SR_DAILY_TOUCH_BONUS) for zp, t in direnc_1d]

    trig = df_trig.sort_values("dt").reset_index(drop=True)
    qqe_bull, qqe_bear, qqe_strength, qqe_trend = compute_qqe(trig["Close"])
    vol_ma = trig["Volume"].rolling(VOLUME_MA_PERIOD).mean()
    volume_spike = (vol_ma > 0) & (trig["Volume"] >= VOLUME_SPIKE_MULT * vol_ma)
    atr_val = atr(trig, ATR_PERIOD)
    atr_pct = atr_val / trig["Close"]
    atr_pct_avg = atr_pct.rolling(ATR_LOOKBACK).mean()
    atr_ok = atr_pct_avg.notna() & (atr_pct >= ATR_MIN_RATIO * atr_pct_avg)

    closes = trig["Close"].values
    n = len(trig)
    kayitlar = []
    last_signal_idx = None
    COOLDOWN_CANDLES = 16  # 4 saat = 16 x 15dk

    for i in range(1, n):
        if not bool(atr_ok.iloc[i]):
            continue
        # Momentum TEYIDI: donus bir onceki mumda olmus, hala gecerli olmali
        strength_check = float(qqe_strength.iloc[i - 1])
        bull_now = (bool(qqe_bull.iloc[i - 1]) and qqe_trend.iloc[i] == 1
                    and strength_check >= QQE_MIN_STRENGTH)
        bear_now = (bool(qqe_bear.iloc[i - 1]) and qqe_trend.iloc[i] == -1
                    and strength_check >= QQE_MIN_STRENGTH)
        if not (bull_now or bear_now):
            continue
        if last_signal_idx is not None and (i - last_signal_idx) < COOLDOWN_CANDLES:
            continue
        price_now = closes[i]

        long_zone = zone_below(price_now, destek_bolgeleri) if bull_now else None
        short_zone = zone_above(price_now, direnc_bolgeleri) if bear_now else None
        if long_zone is None and short_zone is None:
            continue

        strength_now = strength_check
        spike_now = bool(volume_spike.iloc[i])

        long_puan = short_puan = 0
        if long_zone is not None:
            _, touches, dist_pct = long_zone
            long_puan = (puan_bolge_guc(touches) + puan_yakinlik(dist_pct)
                         + puan_momentum(strength_now) + puan_hacim(spike_now))
        if short_zone is not None:
            _, touches, dist_pct = short_zone
            short_puan = (puan_bolge_guc(touches) + puan_yakinlik(dist_pct)
                          + puan_momentum(strength_now) + puan_hacim(spike_now))

        if long_zone is not None and long_puan >= short_puan:
            direction, puan = "LONG", long_puan
            tp_hedef = nearest_above_any(price_now, direnc_bolgeleri)
        elif short_zone is not None:
            direction, puan = "SHORT", short_puan
            tp_hedef = nearest_below_any(price_now, destek_bolgeleri)
        else:
            continue

        if tp_hedef is None:
            continue
        tp_dist_pct = abs(tp_hedef[0] - price_now) / price_now
        if tp_dist_pct < MIN_TP_DISTANCE_PCT:
            continue

        if puan < MIN_SINYAL_PUAN:
            continue

        guven = "YUKSEK" if puan >= ESIK_YUKSEK_GUVEN else "ORTA"

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
        last_signal_idx = i

    if not kayitlar:
        return pd.DataFrame(), True
    return pd.DataFrame(kayitlar), True


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
    print(f"HEDEF YAKALAMA ORANI (sinyalden sonraki {MFE_HORIZON//96} gun icinde):")
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
