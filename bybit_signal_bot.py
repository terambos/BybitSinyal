"""
Adaptif Kripto Sinyal Botu - v3 (Veriye Dayali Iyilestirme)
==============================================================
Onceki backtest bulgusu: "Yuksek Guven" (trend uyumu yuksek) sinyaller,
"Orta Guven"den DAHA KOTU performans gosterdi - yani buyuk zaman dilimi
trend uyumu bu stratejide gercek bir avantaj katmiyor (bazen tam tersi).

BU YUZDEN: Trend (Kalman) puanlamadan tamamen CIKARILDI. Bunun yerine,
gercekten ayirt edici oldugu gorulen 4 faktore agirlik verildi:

ZORUNLU ON KOSUL (ikisi de sart):
  a) QQE Momentum donusu (15dk)
  b) Fiyatin bir Destek/Direnc bolgesine yakin olmasi (4S veriden)

GUVEN PUANI (10 uzerinden, sadece ON KOSUL saglaninca hesaplanir):
  - Bolge GUCU (kac kez test edilmis): maks 3 puan
  - Bolgeye YAKINLIK (ne kadar yakin, ne kadar keskin bir reaksiyon
    bolgesi): maks 3 puan
  - QQE momentum GUCU (RSI'nin kendi bandini ne kadar asarak kestigi -
    zayif/gucuk kesisim ayrimi): maks 2 puan
  - Hacim teyidi: maks 2 puan

Guven esikleri: 8-10 = YUKSEK, 5-7 = ORTA, 5 alti = sinyal yok.

Calistirma: GitHub Actions ile her 15 dakikada bir.
Gerekli ortam degiskenleri (GitHub Secrets):
  GMAIL_ADDRESS, GMAIL_APP_PASSWORD, TO_EMAIL
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
    "LINK", "DOT", "NEAR", "INJ", "OP", "SEI",
    "BNB", "SHIB", "TAO", "ENA", "ATOM", "ALGO",
]

YF_SUFFIX = "-USD"
YF_TICKER_OVERRIDES = {
    "SUI": "SUI20947", "HYPE": "HYPE32196", "APT": "APT21794", "TAO": "TAO22974",
}
TREND_PERIOD = "730d"       # 4S S/R bolgelerini olusturmak icin gerekli
TREND_INTERVAL = "1h"
TRIGGER_PERIOD = "60d"
TRIGGER_INTERVAL = "15m"

# --- QQE (momentum donusu) ---
QQE_RSI_PERIOD = 14
QQE_SMOOTHING = 5
QQE_FACTOR = 4.236
QQE_WILDER_PERIOD = QQE_RSI_PERIOD * 2 - 1

# --- Destek/Direnc bolgesi ---
SR_SWING_ORDER = 3
SR_CLUSTER_TOLERANCE = 0.015
SR_MIN_TOUCHES = 2
SR_PROXIMITY_PCT = 0.015

# --- Hacim ---
VOLUME_MA_PERIOD = 20
VOLUME_SPIKE_MULT = 1.2

# --- Guven puanlama esikleri ---
MAX_PUAN = 10
ESIK_ORTA_GUVEN = 5
ESIK_YUKSEK_GUVEN = 8

STATE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state.json")
TR_TZ = timezone(timedelta(hours=3))


# ----------------------------------------------------------------------
# TEMEL INDIKATORLER
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
    return (100 - (100 / (1 + rs))).fillna(50)


# ----------------------------------------------------------------------
# QQE MOMENTUM DONUSU (+ kesisim GUCU olcumu)
# ----------------------------------------------------------------------

def compute_qqe(close: pd.Series):
    """
    Doner: bullish_cross, bearish_cross (bool serileri) ve
    cross_strength (kesisim anindaki RSI'nin bandi ne kadar astigini,
    bandin genisligine oranla olcen - 0'dan yukari, buyuk = guclu donus).
    """
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

    # Kesisim gucu: RSI'nin, kesilen bandin ne kadar UZAGINA gectigi,
    # bandin genisligine (dar) oranla - buyuk oran = guclu/net donus
    shortband_prev = pd.Series(shortband, index=close.index).shift(1)
    longband_prev = pd.Series(longband, index=close.index).shift(1)
    dar_safe = dar.replace(0, np.nan)
    bull_strength = ((rsi_ma - shortband_prev) / dar_safe).clip(lower=0)
    bear_strength = ((longband_prev - rsi_ma) / dar_safe).clip(lower=0)
    cross_strength = np.where(bullish_cross, bull_strength, np.where(bearish_cross, bear_strength, 0.0))
    cross_strength = pd.Series(cross_strength, index=close.index).fillna(0.0)

    return bullish_cross, bearish_cross, cross_strength


# ----------------------------------------------------------------------
# DESTEK / DIRENC BOLGELERI
# ----------------------------------------------------------------------

def find_sr_zones(df_htf: pd.DataFrame):
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


# ----------------------------------------------------------------------
# PUANLAMA YARDIMCILARI (veriye dayali: bolge gucu + yakinlik + momentum + hacim)
# ----------------------------------------------------------------------

def puan_bolge_guc(touches: int) -> int:
    if touches >= 6:
        return 3
    if touches >= 4:
        return 2
    return 1  # min 2 dokunma zaten gate icin sart


def puan_yakinlik(dist_pct: float) -> int:
    if dist_pct <= 0.005:
        return 3
    if dist_pct <= 0.010:
        return 2
    return 1


def puan_momentum(strength: float) -> int:
    if strength >= 0.5:
        return 2
    if strength >= 0.15:
        return 1
    return 0


def puan_hacim(spike: bool) -> int:
    return 2 if spike else 0


# ----------------------------------------------------------------------
# VERI CEKME
# ----------------------------------------------------------------------

def fetch_history(base: str, period: str, interval: str) -> pd.DataFrame | None:
    ticker = f"{YF_TICKER_OVERRIDES.get(base, base)}{YF_SUFFIX}"
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    except Exception as e:
        print(f"UYARI: {ticker} ({interval}) cekilemedi: {e}", file=sys.stderr)
        return None
    if df is None or df.empty:
        return None
    df = df.reset_index()
    dt_col = "Datetime" if "Datetime" in df.columns else df.columns[0]
    df = df.rename(columns={dt_col: "dt"})
    df["dt"] = pd.to_datetime(df["dt"], utc=True)

    interval_map = {"15m": pd.Timedelta(minutes=15), "1h": pd.Timedelta(hours=1)}
    step = interval_map.get(interval, pd.Timedelta(hours=1))
    now = pd.Timestamp.now(tz="UTC")
    if not df.empty and (df["dt"].iloc[-1] + step) > now:
        df = df.iloc[:-1]

    return df[["dt", "Open", "High", "Low", "Close", "Volume"]].reset_index(drop=True)


def resample(df_1h: pd.DataFrame, rule: str) -> pd.DataFrame:
    df = df_1h.set_index("dt")
    out = df.resample(rule, label="right", closed="right").agg({
        "Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum",
    }).dropna()
    return out.reset_index()


# ----------------------------------------------------------------------
# ANA STRATEJI
# ----------------------------------------------------------------------

def evaluate_symbol(base: str) -> dict | None:
    df_trend_1h = fetch_history(base, TREND_PERIOD, TREND_INTERVAL)
    if df_trend_1h is None or len(df_trend_1h) < 200 * 24:
        return None

    df_4h = resample(df_trend_1h, "4h")
    if len(df_4h) < 60:
        return None

    destek_bolgeleri, direnc_bolgeleri = find_sr_zones(df_4h)

    df_trig = fetch_history(base, TRIGGER_PERIOD, TRIGGER_INTERVAL)
    if df_trig is None or len(df_trig) < 100:
        return None

    qqe_bull, qqe_bear, qqe_strength = compute_qqe(df_trig["Close"])
    vol_ma = df_trig["Volume"].rolling(VOLUME_MA_PERIOD).mean()

    i = len(df_trig) - 1
    price_now = float(df_trig["Close"].iloc[i])
    qqe_bull_now = bool(qqe_bull.iloc[i])
    qqe_bear_now = bool(qqe_bear.iloc[i])
    strength_now = float(qqe_strength.iloc[i])
    vol_now = df_trig["Volume"].iloc[i]
    vol_ma_now = vol_ma.iloc[i]
    volume_spike = bool(vol_ma_now > 0 and vol_now >= VOLUME_SPIKE_MULT * vol_ma_now)

    long_zone = zone_below(price_now, destek_bolgeleri) if qqe_bull_now else None
    short_zone = zone_above(price_now, direnc_bolgeleri) if qqe_bear_now else None

    if long_zone is None and short_zone is None:
        return None

    long_puan = short_puan = 0
    if long_zone is not None:
        _, touches, dist_pct = long_zone
        long_puan = (puan_bolge_guc(touches) + puan_yakinlik(dist_pct)
                     + puan_momentum(strength_now) + puan_hacim(volume_spike))
    if short_zone is not None:
        _, touches, dist_pct = short_zone
        short_puan = (puan_bolge_guc(touches) + puan_yakinlik(dist_pct)
                      + puan_momentum(strength_now) + puan_hacim(volume_spike))

    if long_zone is not None and long_puan >= short_puan:
        direction, puan, bolge = "LONG", long_puan, long_zone
        tp_hedef = nearest_above_any(price_now, direnc_bolgeleri)
    elif short_zone is not None:
        direction, puan, bolge = "SHORT", short_puan, short_zone
        tp_hedef = nearest_below_any(price_now, destek_bolgeleri)
    else:
        return None

    if puan < ESIK_ORTA_GUVEN:
        return None

    guven_etiketi = "YUKSEK" if puan >= ESIK_YUKSEK_GUVEN else "ORTA"
    candle_ts = int(df_trig["dt"].iloc[i].timestamp() * 1000)

    return {
        "symbol": base,
        "direction": direction,
        "puan": puan,
        "guven": guven_etiketi,
        "candle_ts": candle_ts,
        "price": price_now,
        "bolge_fiyat": bolge[0],
        "bolge_dokunma": bolge[1],
        "bolge_uzaklik_%": round(bolge[2] * 100, 2),
        "momentum_gucu": round(strength_now, 2),
        "tp_hedef": tp_hedef[0] if tp_hedef else None,
        "volume_ratio": round(float(vol_now / vol_ma_now), 2) if vol_ma_now else None,
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
    guven_tr = "YUKSEK GUVEN" if sig["guven"] == "YUKSEK" else "ORTA GUVEN"
    subject = f"[SINYAL-{sig['guven']}] {sig['symbol']} {sig['direction']} ({sig['puan']}/{MAX_PUAN})"

    tp_satir = f"Olasi TP (en yakin ters bolge): {sig['tp_hedef']}\n" if sig["tp_hedef"] else ""

    body = (
        f"{sig['symbol']} icin {sig['direction']} sinyali olustu. [{guven_tr}: {sig['puan']}/{MAX_PUAN} puan]\n\n"
        f"Mum kapanis zamani (TR): {dt_tr.strftime('%d.%m.%Y %H:%M')}\n"
        f"Fiyat: {sig['price']}\n"
        f"Yakin {'Destek' if sig['direction']=='LONG' else 'Direnc'} bolgesi: {sig['bolge_fiyat']} "
        f"({sig['bolge_dokunma']} kez test edilmis, %{sig['bolge_uzaklik_%']} uzaklikta)\n"
        f"{tp_satir}"
        f"Momentum gucu (QQE): {sig['momentum_gucu']}\n"
        f"Hacim / 20-ort. oran: {sig['volume_ratio']}\n\n"
        f"On kosul: QQE momentum donusu + fiyatin bir S/R bolgesine yakinligi.\n"
        f"Guven puani: bolge gucu(3) + bolgeye yakinlik(3) + momentum gucu(2) + hacim(2).\n"
        f"8-10 Yuksek, 5-7 Orta guven demektir.\n"
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
            print("TEST MAILI GONDERILDI.")
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
