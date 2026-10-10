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

# ADX (trend gucu) ve anomali/haber-sicramasi filtresi - canli botla ayni
ADX_PERIOD = 14
ADX_MIN = 20
ANOMALY_LOOKBACK = 3
ANOMALY_ATR_MULT = 2.5
SYMBOL_COOLDOWN_MS = 4 * 60 * 60 * 1000
ESIK_YUKSEK_GUVEN = 9   # 9-10 YUKSEK, 8 ORTA (etiket sadece puan bandidir, isabet garantisi degil)
MIN_SINYAL_PUAN = 8   # 9-10'a sikistirma denendi, isabeti artirmadi - 8'e geri donuldu
SIRALI_HEDEF_PCT = 5.0   # sirali sonuc: once +%5'e mi, once -%3'e mi degdi
SIRALI_STOP_PCT = 3.0

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


def adx(df, period=ADX_PERIOD):
    high, low, close = df["High"], df["Low"], df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_ = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (atr_ + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, adjust=False).mean() / (atr_ + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    adx_val = dx.ewm(alpha=1 / period, adjust=False).mean()
    return adx_val, plus_di, minus_di


def obv_rising_series(df, lookback=5):
    obv = (np.sign(df["Close"].diff()).fillna(0) * df["Volume"]).cumsum()
    return obv.diff(lookback) > 0


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


def find_sr_zones(df_htf, swing_order, anchored=True):
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
            ref = (group[0] if anchored else group[-1]) if group else None
            if ref is None or (p - ref) / ref <= SR_CLUSTER_TOLERANCE:
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


def puan_hacim(spike, obv_confirms):
    if spike and obv_confirms:
        return 2
    if spike or obv_confirms:
        return 1
    return 0


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


def bolgeleri_hesapla(df_4h, df_1d, anchored):
    d4, r4 = find_sr_zones(df_4h, SR_SWING_ORDER_4H, anchored)
    destek, direnc = list(d4), list(r4)
    if len(df_1d) >= 2 * SR_SWING_ORDER_1D + 5:
        d1, r1 = find_sr_zones(df_1d, SR_SWING_ORDER_1D, anchored)
        destek += [(zp, t + SR_DAILY_TOUCH_BONUS) for zp, t in d1]
        direnc += [(zp, t + SR_DAILY_TOUCH_BONUS) for zp, t in r1]
    return destek, direnc


def backtest_symbol(base):
    """
    Ayni veri uzerinde IKI yontemi kosturur:
      ESKI : S/R bolgeleri tum gecmisten (gelecek dahil), zincirleme kumeleme, ADX 1 saate kadar ileriye bakar
      YENI : her sinyal icin SADECE o ana kadar TAMAMLANMIS 4S/1G mumlardan bolge, sabit capali kumeleme
             (canli botun gercek davranisi budur)
    Doner: ({"ESKI": df, "YENI": df}, veri_tamam_mi)
    """
    df_trend_1h = fetch_history(base, TREND_PERIOD, TREND_INTERVAL)
    time.sleep(2)
    df_trig = fetch_history(base, TRIGGER_PERIOD, TRIGGER_INTERVAL)

    if df_trend_1h is None or df_trig is None:
        return {}, False
    if len(df_trend_1h) < 200 * 24 or len(df_trig) < 300:
        return {}, False

    df_4h = resample(df_trend_1h, "4h")
    df_1d = resample(df_trend_1h, "1D")
    if len(df_4h) < 60:
        return {}, False

    trig = df_trig.sort_values("dt").reset_index(drop=True)
    qqe_bull, qqe_bear, qqe_strength, qqe_trend = compute_qqe(trig["Close"])
    vol_ma = trig["Volume"].rolling(VOLUME_MA_PERIOD).mean()
    volume_spike = (vol_ma > 0) & (trig["Volume"] >= VOLUME_SPIKE_MULT * vol_ma)
    atr_val = atr(trig, ATR_PERIOD)
    atr_pct = atr_val / trig["Close"]
    atr_pct_avg = atr_pct.rolling(ATR_LOOKBACK).mean()
    atr_ok = atr_pct_avg.notna() & (atr_pct >= ATR_MIN_RATIO * atr_pct_avg)
    obv_up_series = obv_rising_series(trig)
    bar_range = trig["High"] - trig["Low"]
    anomaly_series = bar_range.rolling(ANOMALY_LOOKBACK).max() > (ANOMALY_ATR_MULT * atr_val)

    # ADX: 4S bar etiketi, bardaki SON saatlik mumun baslangicidir -> bar ancak etiket + 1 saat sonra tamamlanir
    adx_4h, plus_di_4h, minus_di_4h = adx(df_4h)
    adx_df = pd.DataFrame({
        "dt": df_4h["dt"].reset_index(drop=True),
        "adx": adx_4h.reset_index(drop=True),
        "plus_di": plus_di_4h.reset_index(drop=True),
        "minus_di": minus_di_4h.reset_index(drop=True),
    })
    saat = pd.Timedelta(hours=1)
    adx_eski = pd.merge_asof(trig[["dt"]], adx_df, on="dt", direction="backward")
    # Karar, 15dk mumun KAPANISINDA verilir (acilis + 15dk): o ana kadar tamamlanmis barlar bilinir
    karar_zamani = trig[["dt"]].assign(dt=trig["dt"] + pd.Timedelta(minutes=15))
    adx_yeni = pd.merge_asof(karar_zamani, adx_df.assign(dt=adx_df["dt"] + saat), on="dt", direction="backward")

    # Noktasal-zaman (ileriye bakissiz) bolgeler icin tamamlanma zamanlari
    avail4 = (df_4h["dt"] + saat).values.astype("datetime64[ns]")
    avail1 = (df_1d["dt"] + saat).values.astype("datetime64[ns]")
    T_degerler = (trig["dt"] + pd.Timedelta(minutes=15)).values.astype("datetime64[ns]")  # karar = mum kapanisi
    onbellek = {}

    def bolge_yeni(T):
        n4 = int(np.searchsorted(avail4, T, side="right"))
        n1 = int(np.searchsorted(avail1, T, side="right"))
        anahtar = (n4, n1)
        if anahtar not in onbellek:
            if n4 < 60:
                onbellek[anahtar] = ([], [])
            else:
                onbellek[anahtar] = bolgeleri_hesapla(df_4h.iloc[:n4], df_1d.iloc[:n1], True)
        return onbellek[anahtar]

    sabit_eski = bolgeleri_hesapla(df_4h, df_1d, False)

    closes = trig["Close"].values
    n = len(trig)
    COOLDOWN_CANDLES = 16  # 4 saat = 16 x 15dk

    def tara(mod):
        adx_m = adx_eski if mod == "ESKI" else adx_yeni
        trend_guclu = (adx_m["adx"].fillna(0) >= ADX_MIN).values
        di_bull = (adx_m["plus_di"].fillna(0) > adx_m["minus_di"].fillna(0)).values
        di_bear = (adx_m["minus_di"].fillna(0) > adx_m["plus_di"].fillna(0)).values
        kayitlar = []
        last_signal_idx = None

        for i in range(1, n):
            if not bool(atr_ok.iloc[i]):
                continue
            if bool(anomaly_series.iloc[i]):
                continue
            strength_check = float(qqe_strength.iloc[i - 1])
            bull_now = (bool(qqe_bull.iloc[i - 1]) and qqe_trend.iloc[i] == 1
                        and strength_check >= QQE_MIN_STRENGTH)
            bear_now = (bool(qqe_bear.iloc[i - 1]) and qqe_trend.iloc[i] == -1
                        and strength_check >= QQE_MIN_STRENGTH)
            bull_now = bull_now and bool(trend_guclu[i]) and bool(di_bull[i])
            bear_now = bear_now and bool(trend_guclu[i]) and bool(di_bear[i])
            if not (bull_now or bear_now):
                continue
            if last_signal_idx is not None and (i - last_signal_idx) < COOLDOWN_CANDLES:
                continue
            price_now = closes[i]

            destek_bolgeleri, direnc_bolgeleri = sabit_eski if mod == "ESKI" else bolge_yeni(T_degerler[i])

            long_zone = zone_below(price_now, destek_bolgeleri) if bull_now else None
            short_zone = zone_above(price_now, direnc_bolgeleri) if bear_now else None
            if long_zone is None and short_zone is None:
                continue

            strength_now = strength_check
            spike_now = bool(volume_spike.iloc[i])
            obv_now = bool(obv_up_series.iloc[i])

            long_puan = short_puan = 0
            if long_zone is not None:
                _, touches, dist_pct = long_zone
                long_puan = (puan_bolge_guc(touches) + puan_yakinlik(dist_pct)
                             + puan_momentum(strength_now) + puan_hacim(spike_now, obv_now))
            if short_zone is not None:
                _, touches, dist_pct = short_zone
                short_puan = (puan_bolge_guc(touches) + puan_yakinlik(dist_pct)
                              + puan_momentum(strength_now) + puan_hacim(spike_now, not obv_now))

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
            if abs(tp_hedef[0] - price_now) / price_now < MIN_TP_DISTANCE_PCT:
                continue
            if puan < MIN_SINYAL_PUAN:
                continue

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

            # SIRALI sonuc: once hedefe mi (+%5), once stop'a mi (-%3) degdi?
            hit = pct_path >= SIRALI_HEDEF_PCT
            stp = pct_path <= -SIRALI_STOP_PCT
            t_hit = int(np.argmax(hit)) if hit.any() else None
            t_stp = int(np.argmax(stp)) if stp.any() else None
            if t_hit is not None and (t_stp is None or t_hit < t_stp):
                sonuc, sirali_getiri = "kazanc", SIRALI_HEDEF_PCT
            elif t_stp is not None:
                sonuc, sirali_getiri = "kayip", -SIRALI_STOP_PCT
            else:
                sonuc, sirali_getiri = "belirsiz", float(pct_path[-1])

            kayit = {
                "mod": mod, "symbol": base, "dt": trig["dt"].iloc[i], "direction": direction,
                "puan": puan, "guven": "YUKSEK" if puan >= ESIK_YUKSEK_GUVEN else "ORTA",
                "Close": price_now, "mfe_%": round(mfe, 2), "mae_%": round(mae, 2),
                "sonuc": sonuc, "sirali_getiri_%": round(sirali_getiri, 2),
            }
            for h in HEDEF_YUZDELER:
                kayit[f"hedef_%{h}_ulasti"] = mfe >= h
            kayitlar.append(kayit)
            last_signal_idx = i

        return pd.DataFrame(kayitlar)

    return {"ESKI": tara("ESKI"), "YENI": tara("YENI")}, True


def rapor(df, baslik):
    print("\n" + "#" * 64)
    print(baslik)
    print("#" * 64)
    if df is None or df.empty:
        print("Sinyal yok.")
        return
    n = len(df)
    print(f"TOPLAM SINYAL: {n}   |   LONG: {(df['direction']=='LONG').sum()}  SHORT: {(df['direction']=='SHORT').sum()}")
    print(f"Puan 9-10: {(df['puan']>=9).sum()}   Puan 8: {(df['puan']==8).sum()}")
    print("\nSembol basina (ilk 10):")
    print(df["symbol"].value_counts().head(10).to_string())

    print(f"\nHEDEFE DEGME ORANI ({MFE_HORIZON//96} gun icinde HERHANGI BIR anda; stop dikkate alinmaz):")
    for h in HEDEF_YUZDELER:
        col = f"hedef_%{h}_ulasti"
        print(f"  %{h}: %{df[col].mean()*100:.1f}  ({int(df[col].sum())}/{n})")
    print(f"Ortalama MFE: %{df['mfe_%'].mean():.2f} | Medyan MFE: %{df['mfe_%'].median():.2f}")
    print(f"Ortalama MAE: %{df['mae_%'].mean():.2f} | Medyan MAE: %{df['mae_%'].median():.2f}")

    kz = (df["sonuc"] == "kazanc").mean() * 100
    ky = (df["sonuc"] == "kayip").mean() * 100
    bl = (df["sonuc"] == "belirsiz").mean() * 100
    basabas = SIRALI_STOP_PCT / (SIRALI_HEDEF_PCT + SIRALI_STOP_PCT) * 100
    print(f"\nSIRALI SONUC (hedef +%{SIRALI_HEDEF_PCT:g} / stop -%{SIRALI_STOP_PCT:g}, hangisine ONCE degdi):")
    print(f"  Kazanc: %{kz:.1f} | Kayip: %{ky:.1f} | Belirsiz (3 gunde ikisine de degmedi): %{bl:.1f}")
    print(f"  Basabas kazanc orani (belirsiz yokken): %{basabas:.1f}")
    print(f"  Ortalama getiri / islem: %{df['sirali_getiri_%'].mean():.2f}  (ucret/kayma HARIC)")

    print("\nPUAN BANDINA GORE:")
    for ad, maske in (("Puan 9-10", df["puan"] >= 9), ("Puan 8", df["puan"] == 8)):
        alt = df[maske]
        if alt.empty:
            print(f"  {ad}: sinyal yok")
            continue
        print(f"  {ad}: {len(alt)} sinyal | %5'e degen: %{alt['hedef_%5_ulasti'].mean()*100:.1f} | "
              f"sirali kazanc: %{(alt['sonuc']=='kazanc').mean()*100:.1f} | ort. getiri: %{alt['sirali_getiri_%'].mean():.2f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tum = {"ESKI": [], "YENI": []}
    basarili, alinamayan = [], []

    for idx, base in enumerate(CRYPTO_SYMBOLS):
        print(f"[{idx+1}/{len(CRYPTO_SYMBOLS)}] Taraniyor: {base}...")
        try:
            sonuclar, veri_tamam = backtest_symbol(base)
        except Exception as e:
            print(f"HATA ({base}): {e}", file=sys.stderr)
            alinamayan.append(base)
            continue

        if not veri_tamam:
            print("  -> VERI ALINAMADI")
            alinamayan.append(base)
        else:
            basarili.append(base)
            for mod in ("ESKI", "YENI"):
                if not sonuclar[mod].empty:
                    tum[mod].append(sonuclar[mod])
            print(f"  -> eski yontem: {len(sonuclar['ESKI'])} | yeni yontem: {len(sonuclar['YENI'])} sinyal")

        if idx < len(CRYPTO_SYMBOLS) - 1:
            time.sleep(3)

    print("\n" + "=" * 64)
    print(f"VERI DURUMU: {len(basarili)}/{len(CRYPTO_SYMBOLS)} coin basariyla tarandi")
    if alinamayan:
        print(f"Veri alinamayan coinler: {', '.join(alinamayan)}")

    df_eski = pd.concat(tum["ESKI"], ignore_index=True) if tum["ESKI"] else pd.DataFrame()
    df_yeni = pd.concat(tum["YENI"], ignore_index=True) if tum["YENI"] else pd.DataFrame()

    rapor(df_eski, "ESKI YONTEM (tum gecmisten bolge + zincirleme kumeleme = ileriye bakis VAR)")
    rapor(df_yeni, "YENI YONTEM (sadece o ana kadar tamamlanmis mumlar + sabit capali kumeleme = CANLI BOT BU)")

    birlesik = pd.concat([d for d in (df_eski, df_yeni) if not d.empty], ignore_index=True) \
        if (not df_eski.empty or not df_yeni.empty) else pd.DataFrame()
    if not birlesik.empty:
        out_path = os.path.join(OUT_DIR, "backtest_sonuclari.csv")
        birlesik.sort_values(["mod", "dt"]).to_csv(out_path, index=False)
        print(f"\nTum detaylar kaydedildi: {out_path}")

    print("\nNOT: Gercek performansi YENI YONTEM yansitir. ESKI ile arasindaki fark, ileriye bakisin ne kadar iyimserlik kattigini gosterir.")


if __name__ == "__main__":
    main()
