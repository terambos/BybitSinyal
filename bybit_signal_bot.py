"""
Coklu Zaman Dilimi Kripto Sinyal Botu (Yahoo Finance veri kaynagi)
====================================================================
Binance ve Bybit'in genel API'leri GitHub Actions gibi bulut sunuculardan
gelen istekleri engelledigi icin veri kaynagi olarak Yahoo Finance
kullaniliyor (ucretsiz, API anahtari gerektirmiyor, bulut IP'lerini
engellemiyor). Sen yine istedigin borsada - orn. Bybit'te - islem
acabilirsin, buyuk coinlerin fiyatlari borsalar arasi pratikte ayni
hareket eder.

Strateji:
  - Trend filtresi: 1G (EMA200) + 4S (EMA50 > EMA200)
  - Tetikleyici (1S kapanisinda): RSI(14) 35 ustune/65 altina kesisi
                                   + MACD kesisimi
                                   + Hacim patlamasi (SMA20'nin 1.5 kati)
                                   + ATR volatilite filtresi
  - Trend YUKARI + tum LONG kosullari -> LONG sinyali
  - Trend ASAGI + tum SHORT kosullari -> SHORT sinyali

Tum zaman dilimleri (1S, 4S, 1G) tek bir saatlik veri cekiminden
turetiliyor (4S ve 1G, saatlik mumlardan yeniden ornekleniyor).

Ayni sinyali tekrar tekrar mail atmamak icin state.json'da hafiza tutar.

Calistirma: GitHub Actions ile saatte bir (bkz. .github/workflows/signal-bot.yml)
Gerekli ortam degiskenleri (GitHub Secrets):
  GMAIL_ADDRESS        -> gonderen gmail adresi
  GMAIL_APP_PASSWORD   -> 16 haneli Gmail uygulama sifresi
  TO_EMAIL             -> sinyallerin gidecegi adres
"""

import yfinance as yf
import pandas as pd
import numpy as np
import smtplib
import json
import os
import sys
import warnings
from email.mime.text import MIMEText
from datetime import datetime, timezone, timedelta

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# AYARLAR
# ----------------------------------------------------------------------

CRYPTO_SYMBOLS = [
    "BTC", "ETH", "XRP", "SUI", "HYPE", "AVAX", "ARB", "APT",
    "FET", "RENDER", "TIA", "DOGE", "SOL",
    # Kaliteli/hacimli ek coinler
    "LINK", "DOT", "NEAR", "INJ", "OP", "SEI",
]

YF_SUFFIX = "-USD"          # Yahoo Finance kripto sembol formati (orn. BTC-USD)
HISTORY_PERIOD = "730d"     # 1 saatlik veri icin Yahoo'nun izin verdigi max araligi

RSI_PERIOD = 14
RSI_LONG_LEVEL = 35
RSI_SHORT_LEVEL = 65
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULT = 1.5
ATR_PERIOD = 14
ATR_LOOKBACK = 50
ATR_MIN_RATIO = 0.5

EMA_1D = 200
EMA_4H_FAST = 50
EMA_4H_SLOW = 200

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
TR_TZ = timezone(timedelta(hours=3))


# ----------------------------------------------------------------------
# INDIKATOR HESAPLAMALARI
# ----------------------------------------------------------------------

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    out = 100 - (100 / (1 + rs))
    return out.fillna(50)


def macd(series: pd.Series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    return macd_line, signal_line


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["High"], df["Low"], df["Close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


# ----------------------------------------------------------------------
# VERI CEKME (YAHOO FINANCE)
# ----------------------------------------------------------------------

def fetch_hourly_history(base: str) -> pd.DataFrame | None:
    """Bir coin icin saatlik OHLCV verisini ceker. Bulamazsa None doner."""
    ticker = f"{base}{YF_SUFFIX}"
    try:
        df = yf.Ticker(ticker).history(period=HISTORY_PERIOD, interval="1h", auto_adjust=False)
    except Exception as e:
        print(f"UYARI: {ticker} icin veri cekilemedi: {e}", file=sys.stderr)
        return None

    if df is None or df.empty or len(df) < ATR_LOOKBACK + RSI_PERIOD + 10:
        return None

    df = df.reset_index()
    dt_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={dt_col: "dt"})
    df["dt"] = pd.to_datetime(df["dt"], utc=True)

    now = pd.Timestamp.now(tz="UTC")
    if not df.empty and (df["dt"].iloc[-1] + pd.Timedelta(hours=1)) > now:
        df = df.iloc[:-1]

    return df[["dt", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)


def resample(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df_1h.set_index("dt")
    out = df.resample(rule, label="right", closed="right").agg({
        "Open": "first",
        "High": "max",
        "Low": "min",
        "Close": "last",
        "Volume": "sum",
    }).dropna()
    return out.reset_index()


# ----------------------------------------------------------------------
# STRATEJI
# ----------------------------------------------------------------------

def evaluate_symbol(base: str) -> dict | None:
    df_1h = fetch_hourly_history(base)
    if df_1h is None:
        return None

    df_4h = resample(df_1h, "4h")
    df_1d = resample(df_1h, "1D")

    if len(df_1d) < EMA_1D or len(df_4h) < EMA_4H_SLOW or len(df_1h) < ATR_LOOKBACK + 2:
        return None

    ema200_1d = ema(df_1d["Close"], EMA_1D)
    close_1d = df_1d["Close"].iloc[-1]
    trend_up_1d = close_1d > ema200_1d.iloc[-1]
    trend_down_1d = close_1d < ema200_1d.iloc[-1]

    ema50_4h = ema(df_4h["Close"], EMA_4H_FAST)
    ema200_4h = ema(df_4h["Close"], EMA_4H_SLOW)
    close_4h = df_4h["Close"].iloc[-1]
    trend_up_4h = (ema50_4h.iloc[-1] > ema200_4h.iloc[-1]) and (close_4h > ema50_4h.iloc[-1])
    trend_down_4h = (ema50_4h.iloc[-1] < ema200_4h.iloc[-1]) and (close_4h < ema50_4h.iloc[-1])

    trend_up = trend_up_1d and trend_up_4h
    trend_down = trend_down_1d and trend_down_4h

    if not (trend_up or trend_down):
        return None

    rsi_1h = rsi(df_1h["Close"], RSI_PERIOD)
    macd_line, signal_line = macd(df_1h["Close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    vol_ma = df_1h["Volume"].rolling(VOLUME_MA_PERIOD).mean()
    atr_1h = atr(df_1h, ATR_PERIOD)
    atr_pct = atr_1h / df_1h["Close"]
    atr_pct_avg = atr_pct.rolling(ATR_LOOKBACK).mean()

    i = len(df_1h) - 1
    rsi_now, rsi_prev = rsi_1h.iloc[i], rsi_1h.iloc[i - 1]
    macd_now, macd_prev = macd_line.iloc[i], macd_line.iloc[i - 1]
    sig_now, sig_prev = signal_line.iloc[i], signal_line.iloc[i - 1]
    vol_now = df_1h["Volume"].iloc[i]
    vol_ma_now = vol_ma.iloc[i]

    rsi_cross_up = (rsi_prev < RSI_LONG_LEVEL) and (rsi_now >= RSI_LONG_LEVEL)
    rsi_cross_down = (rsi_prev > RSI_SHORT_LEVEL) and (rsi_now <= RSI_SHORT_LEVEL)
    macd_cross_up = (macd_prev < sig_prev) and (macd_now >= sig_now)
    macd_cross_down = (macd_prev > sig_prev) and (macd_now <= sig_now)
    volume_spike = vol_ma_now > 0 and vol_now >= VOLUME_SPIKE_MULT * vol_ma_now
    atr_ok = pd.notna(atr_pct_avg.iloc[i]) and atr_pct.iloc[i] >= ATR_MIN_RATIO * atr_pct_avg.iloc[i]

    direction = None
    if trend_up and rsi_cross_up and macd_cross_up and volume_spike and atr_ok:
        direction = "LONG"
    elif trend_down and rsi_cross_down and macd_cross_down and volume_spike and atr_ok:
        direction = "SHORT"

    if direction is None:
        return None

    candle_ts = int(df_1h["dt"].iloc[i].timestamp() * 1000)

    return {
        "symbol": base,
        "direction": direction,
        "candle_ts": candle_ts,
        "price": float(df_1h["Close"].iloc[i]),
        "rsi": round(float(rsi_now), 1),
        "macd": round(float(macd_now), 6),
        "signal_line": round(float(sig_now), 6),
        "volume_ratio": round(float(vol_now / vol_ma_now), 2) if vol_ma_now else None,
        "atr_pct": round(float(atr_pct.iloc[i]) * 100, 2),
    }


# ----------------------------------------------------------------------
# DURUM (DEDUPE) YONETIMI
# ----------------------------------------------------------------------

def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {}


def save_state(state: dict) -> None:
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


# ----------------------------------------------------------------------
# MAIL GONDERME
# ----------------------------------------------------------------------

def send_email(subject: str, body: str) -> None:
    gmail_address = os.environ["GMAIL_ADDRESS"]
    gmail_password = os.environ["GMAIL_APP_PASSWORD"]
    to_email = os.environ["TO_EMAIL"]

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = to_email

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(gmail_address, gmail_password)
        server.sendmail(gmail_address, [to_email], msg.as_string())


def format_signal_email(sig: dict) -> tuple[str, str]:
    dt_tr = datetime.fromtimestamp(sig["candle_ts"] / 1000, tz=TR_TZ)
    subject = f"[SINYAL] {sig['symbol']} {sig['direction']} - 1S kapanis"
    body = (
        f"{sig['symbol']} icin {sig['direction']} sinyali olustu.\n\n"
        f"Mum kapanis zamani (TR): {dt_tr.strftime('%d.%m.%Y %H:%M')}\n"
        f"Fiyat: {sig['price']}\n"
        f"RSI(14): {sig['rsi']}\n"
        f"MACD: {sig['macd']}  |  Sinyal cizgisi: {sig['signal_line']}\n"
        f"Hacim / 20-ort. oran: {sig['volume_ratio']}\n"
        f"ATR: %{sig['atr_pct']}\n\n"
        f"Kosullar: 1G+4S trend teyidi, 1S RSI/MACD kesisimi, hacim patlamasi, ATR filtresi.\n"
        f"Veri kaynagi: Yahoo Finance. Bu otomatik bir sinyaldir, yatirim tavsiyesi degildir.\n"
    )
    return subject, body


# ----------------------------------------------------------------------
# ANA AKIS
# ----------------------------------------------------------------------

def main():
    print(f"Toplam taranan sembol sayisi: {len(CRYPTO_SYMBOLS)}")

    if os.environ.get("TEST_EMAIL", "").strip().lower() == "true":
        print("TEST MODU: deneme maili gonderiliyor...")
        try:
            send_email(
                "[TEST] Sinyal Botu Mail Kontrolu",
                "Bu bir test mailidir. Bu maili aldiysan, mail gonderim sistemi "
                "calisiyor demektir. Gercek sinyaller ayni formatta gelecek.\n"
            )
            print("TEST MAILI GONDERILDI. Gelen kutunu (ve spam klasorunu) kontrol et.")
        except Exception as e:
            print(f"TEST MAIL HATASI: {e}", file=sys.stderr)
        return

    state = load_state()
    new_signals = []

    for base in CRYPTO_SYMBOLS:
        try:
            result = evaluate_symbol(base)
        except Exception as e:
            print(f"HATA ({base}): {e}", file=sys.stderr)
            continue

        if result is None:
            continue

        key = f"{result['symbol']}::{result['direction']}"
        last_sent_ts = state.get(key)

        if last_sent_ts == result["candle_ts"]:
            continue

        new_signals.append(result)
        state[key] = result["candle_ts"]

    for sig in new_signals:
        subject, body = format_signal_email(sig)
        try:
            send_email(subject, body)
            print(f"Mail gonderildi: {subject}")
        except Exception as e:
            print(f"MAIL HATASI ({sig['symbol']}): {e}", file=sys.stderr)

    save_state(state)
    print(f"Tamamlandi. {len(new_signals)} yeni sinyal bulundu.")


if __name__ == "__main__":
    main()
