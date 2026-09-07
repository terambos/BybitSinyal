"""
Coklu Zaman Dilimi Kripto Sinyal Botu
=====================================
Veri kaynagi: Binance USDT-M Futures (Bybit'in genel API'si bulut/GitHub
Actions IP araliklarini engelledigi icin veri buradan cekiliyor; sen yine de
istedigin borsada - orn. Bybit'te - islem acabilirsin, buyuk coinlerin
fiyatlari borsalar arasi pratikte ayni hareket eder).

Strateji:
  - Trend filtresi: 1G (EMA200) + 4S (EMA50 > EMA200)
  - Tetikleyici (1S kapanisinda): RSI(14) 35 ustune/65 altina kesisi
                                   + MACD kesisimi
                                   + Hacim patlamasi (SMA20'nin 1.5 kati)
                                   + ATR volatilite filtresi
  - Trend YUKARI + tum LONG kosullari -> LONG sinyali
  - Trend ASAGI + tum SHORT kosullari -> SHORT sinyali

Her calistiginda sadece en son KAPANMIS 1 saatlik mumu kontrol eder.
Ayni sinyali tekrar tekrar mail atmamak icin state.json'da hafiza tutar.

Calistirma: GitHub Actions ile saatte bir (bkz. .github/workflows/signal-bot.yml)
Gerekli ortam degiskenleri (GitHub Secrets):
  GMAIL_ADDRESS        -> gonderen gmail adresi
  GMAIL_APP_PASSWORD   -> 16 haneli Gmail uygulama sifresi
  TO_EMAIL             -> sinyallerin gidecegi adres
"""

import ccxt
import pandas as pd
import numpy as np
import smtplib
import json
import os
import sys
from email.mime.text import MIMEText
from datetime import datetime, timezone, timedelta

# ----------------------------------------------------------------------
# AYARLAR
# ----------------------------------------------------------------------

CRYPTO_SYMBOLS = [
    "BTC", "ETH", "XRP", "SUI", "HYPE", "AVAX", "ARB", "APT",
    "FET", "RENDER", "TIA", "DOGE", "SOL",
    # Kaliteli/hacimli ek coinler
    "LINK", "DOT", "NEAR", "INJ", "OP", "SEI",
]

QUOTE = "USDT"

RSI_PERIOD = 14
RSI_LONG_LEVEL = 35     # RSI bu seviyenin USTUNE kesisirse LONG tetikleyici
RSI_SHORT_LEVEL = 65    # RSI bu seviyenin ALTINA kesisirse SHORT tetikleyici
MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULT = 1.5
ATR_PERIOD = 14
ATR_LOOKBACK = 50
ATR_MIN_RATIO = 0.5     # ATR%, kendi 50 donemlik ortalamasinin en az yarisi olmali

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
    high, low, close = df["high"], df["low"], df["close"]
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

def build_exchange() -> ccxt.binanceusdm:
    return ccxt.binanceusdm({
        "enableRateLimit": True,
    })


def get_available_symbols(exchange: ccxt.binanceusdm, bases: list[str]) -> list[str]:
    """Binance USDT-M Futures piyasasinda gercekten islem goren sembolleri dondurur."""
    markets = exchange.markets if exchange.markets else exchange.load_markets()
    available = []
    for base in bases:
        unified = f"{base}/{QUOTE}:{QUOTE}"
        if unified in markets and markets[unified].get("active", True):
            available.append(unified)
    return available


def fetch_closed_ohlcv(exchange: ccxt.binanceusdm, symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """OHLCV ceker ve henuz kapanmamis son mumu atar."""
    raw = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit + 1)
    df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    now_ms = exchange.milliseconds()
    if not df.empty and df["ts"].iloc[-1] + tf_ms > now_ms:
        df = df.iloc[:-1]
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------
# STRATEJI
# ----------------------------------------------------------------------

def evaluate_symbol(exchange: ccxt.binanceusdm, symbol: str) -> dict | None:
    df_1d = fetch_closed_ohlcv(exchange, symbol, "1d", EMA_1D + 5)
    df_4h = fetch_closed_ohlcv(exchange, symbol, "4h", EMA_4H_SLOW + 5)
    df_1h = fetch_closed_ohlcv(exchange, symbol, "1h", max(VOLUME_MA_PERIOD, ATR_LOOKBACK) + RSI_PERIOD + 10)

    if len(df_1d) < EMA_1D or len(df_4h) < EMA_4H_SLOW or len(df_1h) < ATR_LOOKBACK + 2:
        return None  # yeterli gecmis veri yok, atla

    # --- 1G trend ---
    ema200_1d = ema(df_1d["close"], EMA_1D)
    close_1d = df_1d["close"].iloc[-1]
    trend_up_1d = close_1d > ema200_1d.iloc[-1]
    trend_down_1d = close_1d < ema200_1d.iloc[-1]

    # --- 4S trend ---
    ema50_4h = ema(df_4h["close"], EMA_4H_FAST)
    ema200_4h = ema(df_4h["close"], EMA_4H_SLOW)
    close_4h = df_4h["close"].iloc[-1]
    trend_up_4h = (ema50_4h.iloc[-1] > ema200_4h.iloc[-1]) and (close_4h > ema50_4h.iloc[-1])
    trend_down_4h = (ema50_4h.iloc[-1] < ema200_4h.iloc[-1]) and (close_4h < ema50_4h.iloc[-1])

    trend_up = trend_up_1d and trend_up_4h
    trend_down = trend_down_1d and trend_down_4h

    if not (trend_up or trend_down):
        return None  # trend belirsiz, sinyal aranmaz

    # --- 1S tetikleyici ---
    rsi_1h = rsi(df_1h["close"], RSI_PERIOD)
    macd_line, signal_line = macd(df_1h["close"], MACD_FAST, MACD_SLOW, MACD_SIGNAL)
    vol_ma = df_1h["volume"].rolling(VOLUME_MA_PERIOD).mean()
    atr_1h = atr(df_1h, ATR_PERIOD)
    atr_pct = atr_1h / df_1h["close"]
    atr_pct_avg = atr_pct.rolling(ATR_LOOKBACK).mean()

    i = len(df_1h) - 1  # en son kapanmis mum
    rsi_now, rsi_prev = rsi_1h.iloc[i], rsi_1h.iloc[i - 1]
    macd_now, macd_prev = macd_line.iloc[i], macd_line.iloc[i - 1]
    sig_now, sig_prev = signal_line.iloc[i], signal_line.iloc[i - 1]
    vol_now = df_1h["volume"].iloc[i]
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

    candle_ts = int(df_1h["ts"].iloc[i])

    return {
        "symbol": symbol,
        "direction": direction,
        "candle_ts": candle_ts,
        "price": float(df_1h["close"].iloc[i]),
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
    base = sig["symbol"].split("/")[0]
    dt_tr = datetime.fromtimestamp(sig["candle_ts"] / 1000, tz=TR_TZ)
    subject = f"[SINYAL] {base} {sig['direction']} - 1S kapanis"
    body = (
        f"{base} icin {sig['direction']} sinyali olustu.\n\n"
        f"Mum kapanis zamani (TR): {dt_tr.strftime('%d.%m.%Y %H:%M')}\n"
        f"Fiyat: {sig['price']}\n"
        f"RSI(14): {sig['rsi']}\n"
        f"MACD: {sig['macd']}  |  Sinyal cizgisi: {sig['signal_line']}\n"
        f"Hacim / 20-ort. oran: {sig['volume_ratio']}\n"
        f"ATR: %{sig['atr_pct']}\n\n"
        f"Kosullar: 1G+4S trend teyidi, 1S RSI/MACD kesisimi, hacim patlamasi, ATR filtresi.\n"
        f"Bu otomatik bir sinyaldir, yatirim tavsiyesi degildir.\n"
    )
    return subject, body


# ----------------------------------------------------------------------
# ANA AKIS
# ----------------------------------------------------------------------

def main():
    exchange = build_exchange()

    print("Binance sembolleri kontrol ediliyor...")
    crypto_syms = get_available_symbols(exchange, CRYPTO_SYMBOLS)

    missing_crypto = set(CRYPTO_SYMBOLS) - {s.split("/")[0] for s in crypto_syms}
    if missing_crypto:
        print(f"UYARI: Binance'te bulunamayan/aktif olmayan coinler atlandi: {sorted(missing_crypto)}")

    all_symbols = crypto_syms
    print(f"Toplam taranan sembol sayisi: {len(all_symbols)}")

    state = load_state()
    new_signals = []

    for symbol in all_symbols:
        try:
            result = evaluate_symbol(exchange, symbol)
        except Exception as e:
            print(f"HATA ({symbol}): {e}", file=sys.stderr)
            continue

        if result is None:
            continue

        key = f"{result['symbol']}::{result['direction']}"
        last_sent_ts = state.get(key)

        if last_sent_ts == result["candle_ts"]:
            continue  # bu mum icin zaten mail atildi

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
