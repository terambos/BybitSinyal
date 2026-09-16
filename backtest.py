"""
Backtest Scripti
=================
Canli botla (bybit_signal_bot.py) BIREBIR AYNI strateji mantigini gecmis
veriye uygular: her coin icin tum gecmiste ne zaman LONG/SHORT sinyali
olustugunu bulur, sonra sinyalden sonraki 1s / 4s / 24s / 3 gun icinde
fiyatin ne kadar hareket ettigine bakar.

ONEMLI KISITLAMA: Tetikleyici (RSI/MACD/hacim/ATR) 15 dakikalik mumlar
uzerinden calisiyor, Yahoo Finance 15 dakikalik veriyi sadece SON 60 GUN
icin veriyor. Yani bu backtest sadece son ~60 gunu kapsayabiliyor - uzun
vadeli (orn. 1 yillik) bir test degil. Trend (1G/4S) hesaplamasi icin
730 gunluk saatlik veri kullaniliyor, o kisim daha uzun donemi kapsiyor.

Sonuc: results/backtest_sonuclari.csv (her sinyalin detayi) ve konsola
ozet istatistikler yazdirilir.
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

RSI_PERIOD = 14
RSI_LONG_LEVEL = 40       # (eskisi: 35) - hafif gevsetildi
RSI_SHORT_LEVEL = 60      # (eskisi: 65)
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULT = 1.2   # (eskisi: 1.5)
ATR_PERIOD = 14
ATR_LOOKBACK = 50
ATR_MIN_RATIO = 0.5

EMA_1D = 200
EMA_4H_FAST = 50
EMA_4H_SLOW = 200

# Sinyal sonrasi kac 15dk'lik mum ileriye bakilacak (1s=4, 4s=16, 24s=96, 3gun=288)
HORIZONS = {"1s": 4, "4s": 16, "24s": 96, "3gun": 288}

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ----------------------------------------------------------------------
# INDIKATORLER (bybit_signal_bot.py ile ayni)
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


def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def atr(df, period=14):
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ----------------------------------------------------------------------
# VERI CEKME
# ----------------------------------------------------------------------

def fetch_history(base, period, interval, retries=4):
    """
    Veriyi ceker. Yahoo Finance cok sik istek gelince (rate limit) bos cevap
    dondurdugu icin, basarisiz olursa gittikce artan surelerle tekrar dener.
    """
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
            wait = 5 * (attempt + 1)  # 5s, 10s, 15s
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
# TUM GECMIS ICIN SINYAL TARAMASI (vektorel)
# ----------------------------------------------------------------------

def backtest_symbol(base: str):
    """
    Geri donus: (sinyaller_df, veri_tamam_mi)
    veri_tamam_mi False ise Yahoo veri vermemis demektir (sinyal yoklugundan farkli).
    """
    df_trend_1h = fetch_history(base, TREND_PERIOD, TREND_INTERVAL)
    time.sleep(2)  # iki istek arasi kisa bekleme
    df_trig = fetch_history(base, TRIGGER_PERIOD, TRIGGER_INTERVAL)

    if df_trend_1h is None or df_trig is None:
        return pd.DataFrame(), False
    if len(df_trend_1h) < EMA_1D * 24 or len(df_trig) < ATR_LOOKBACK + RSI_PERIOD + 10:
        return pd.DataFrame(), False

    df_4h = resample(df_trend_1h, "4h")
    df_1d = resample(df_trend_1h, "1D")
    if len(df_1d) < EMA_1D or len(df_4h) < EMA_4H_SLOW:
        return pd.DataFrame(), False

    df_1d = df_1d.copy()
    df_1d["ema200"] = ema(df_1d["Close"], EMA_1D)
    df_1d["trend_up_1d"] = df_1d["Close"] > df_1d["ema200"]
    df_1d["trend_down_1d"] = df_1d["Close"] < df_1d["ema200"]

    df_4h = df_4h.copy()
    df_4h["ema50"] = ema(df_4h["Close"], EMA_4H_FAST)
    df_4h["ema200"] = ema(df_4h["Close"], EMA_4H_SLOW)
    df_4h["trend_up_4h"] = (df_4h["ema50"] > df_4h["ema200"]) & (df_4h["Close"] > df_4h["ema50"])
    df_4h["trend_down_4h"] = (df_4h["ema50"] < df_4h["ema200"]) & (df_4h["Close"] < df_4h["ema50"])

    # Trigger zaman damgalarina, o ana kadar bilinen en son 1G/4S trendini esle
    trig = df_trig.sort_values("dt").reset_index(drop=True)
    trig = pd.merge_asof(trig, df_1d[["dt", "trend_up_1d", "trend_down_1d"]], on="dt", direction="backward")
    trig = pd.merge_asof(trig, df_4h[["dt", "trend_up_4h", "trend_down_4h"]], on="dt", direction="backward")
    trig["trend_up"] = trig["trend_up_1d"].fillna(False) & trig["trend_up_4h"].fillna(False)
    trig["trend_down"] = trig["trend_down_1d"].fillna(False) & trig["trend_down_4h"].fillna(False)

    trig["rsi"] = rsi(trig["Close"], RSI_PERIOD)
    macd_line, signal_line = macd(trig["Close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    trig["macd"] = macd_line
    trig["macd_signal"] = signal_line
    trig["vol_ma"] = trig["Volume"].rolling(VOLUME_MA_PERIOD).mean()
    trig["atr"] = atr(trig, ATR_PERIOD)
    trig["atr_pct"] = trig["atr"] / trig["Close"]
    trig["atr_pct_avg"] = trig["atr_pct"].rolling(ATR_LOOKBACK).mean()

    rsi_prev = trig["rsi"].shift(1)
    macd_prev = trig["macd"].shift(1)
    sig_prev = trig["macd_signal"].shift(1)

    rsi_cross_up = (rsi_prev < RSI_LONG_LEVEL) & (trig["rsi"] >= RSI_LONG_LEVEL)
    rsi_cross_down = (rsi_prev > RSI_SHORT_LEVEL) & (trig["rsi"] <= RSI_SHORT_LEVEL)
    macd_cross_up = (macd_prev < sig_prev) & (trig["macd"] >= trig["macd_signal"])
    macd_cross_down = (macd_prev > sig_prev) & (trig["macd"] <= trig["macd_signal"])
    volume_spike = (trig["vol_ma"] > 0) & (trig["Volume"] >= VOLUME_SPIKE_MULT * trig["vol_ma"])
    atr_ok = trig["atr_pct_avg"].notna() & (trig["atr_pct"] >= ATR_MIN_RATIO * trig["atr_pct_avg"])

    long_signal = trig["trend_up"] & rsi_cross_up & macd_cross_up & volume_spike & atr_ok
    short_signal = trig["trend_down"] & rsi_cross_down & macd_cross_down & volume_spike & atr_ok

    trig["direction"] = np.where(long_signal, "LONG", np.where(short_signal, "SHORT", None))
    signals = trig[trig["direction"].notna()].copy()
    signals["symbol"] = base

    # Ileriye donuk getiriler
    closes = trig["Close"].values
    n = len(trig)
    for label, steps in HORIZONS.items():
        future_idx = signals.index + steps
        valid = future_idx < n
        col = np.full(len(signals), np.nan)
        entry_prices = signals["Close"].values
        fut_prices = np.where(valid, closes[np.clip(future_idx, 0, n - 1)], np.nan)
        ret_long = (fut_prices / entry_prices - 1) * 100
        ret = np.where(signals["direction"].values == "LONG", ret_long, -ret_long)
        signals[f"getiri_{label}_%"] = np.round(ret, 2)

    return signals[["symbol", "dt", "direction", "Close"] + [f"getiri_{h}_%" for h in HORIZONS]], True


# ----------------------------------------------------------------------
# ANA AKIS
# ----------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    all_signals = []
    basarili_coinler = []
    veri_alinamayan = []

    for idx, base in enumerate(CRYPTO_SYMBOLS):
        print(f"\n[{idx+1}/{len(CRYPTO_SYMBOLS)}] Taraniyor: {base}...")
        try:
            res, veri_tamam = backtest_symbol(base)
        except Exception as e:
            print(f"HATA ({base}): {e}", file=sys.stderr)
            veri_alinamayan.append(base)
            continue

        if not veri_tamam:
            print(f"  -> VERI ALINAMADI (Yahoo cevap vermedi veya yetersiz gecmis)")
            veri_alinamayan.append(base)
        else:
            basarili_coinler.append(base)
            if res is not None and not res.empty:
                all_signals.append(res)
                print(f"  -> {len(res)} sinyal bulundu")
            else:
                print(f"  -> veri tamam, ama hic sinyal olusmamis")

        # Yahoo'yu yormamak icin coinler arasi bekleme
        if idx < len(CRYPTO_SYMBOLS) - 1:
            time.sleep(3)

    print("\n" + "=" * 60)
    print(f"VERI DURUMU: {len(basarili_coinler)}/{len(CRYPTO_SYMBOLS)} coin basariyla tarandi")
    if veri_alinamayan:
        print(f"Veri alinamayan coinler: {', '.join(veri_alinamayan)}")
    print("=" * 60)

    if not all_signals:
        print("\nHICBIR SINYAL BULUNAMADI. Son 60 gunde strateji hicbir coin icin tetiklenmemis.")
        return

    df = pd.concat(all_signals, ignore_index=True)
    df = df.sort_values("dt")
    out_path = os.path.join(OUT_DIR, "backtest_sonuclari.csv")
    df.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"TOPLAM SINYAL SAYISI: {len(df)}")
    print(f"LONG: {(df['direction']=='LONG').sum()}  |  SHORT: {(df['direction']=='SHORT').sum()}")
    print("=" * 60)

    print("\nSembol basina sinyal sayisi:")
    print(df["symbol"].value_counts().to_string())

    print("\n" + "-" * 60)
    print("ZAMAN DILIMINE GORE PERFORMANS (fees/slippage haric, basit ortalama):")
    print("-" * 60)
    for label in HORIZONS:
        col = f"getiri_{label}_%"
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        win_rate = (valid > 0).mean() * 100
        avg_ret = valid.mean()
        print(f"{label:>5}: {len(valid)} sinyal degerlendirildi | "
              f"Kazanma orani: %{win_rate:.1f} | Ortalama getiri: %{avg_ret:.2f}")

    print("\n" + "-" * 60)
    print("EN IYI 5 VE EN KOTU 5 SINYAL (24s getiriye gore):")
    print("-" * 60)
    valid_24 = df.dropna(subset=["getiri_24s_%"]).sort_values("getiri_24s_%")
    if not valid_24.empty:
        print("\nEn kotu 5:")
        print(valid_24.head(5)[["symbol", "dt", "direction", "getiri_24s_%"]].to_string(index=False))
        print("\nEn iyi 5:")
        print(valid_24.tail(5)[["symbol", "dt", "direction", "getiri_24s_%"]].to_string(index=False))

    print(f"\nTum detaylar kaydedildi: {out_path}")
    print("Bu dosyayi GitHub Actions calismasinin 'Artifacts' bolumunden indirebilirsin.")


if __name__ == "__main__":
    main()
