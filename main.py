from fastapi import FastAPI
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

app = FastAPI(title="NIFTY AI Backend V2.3")


def fetch_yahoo(range_value="1d", interval="5m"):
    url = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI"

    params = {
        "range": range_value,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits"
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    r = requests.get(url, params=params, headers=headers, timeout=15)
    r.raise_for_status()

    data = r.json()["chart"]["result"][0]

    timestamps = data.get("timestamp", [])
    quote = data["indicators"]["quote"][0]

    df = pd.DataFrame({
        "time": pd.to_datetime(timestamps, unit="s"),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", [])
    })

    df = df.dropna(subset=["close"]).reset_index(drop=True)

    return df, data.get("meta", {})


def rsi(series, period=14):
    delta = series.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    return 100 - (100 / (1 + rs))


def calculate_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()

    histogram = macd - signal

    return macd.iloc[-1], signal.iloc[-1], histogram.iloc[-1]


def trend_from_data(df):
    if len(df) < 20:
        return "UNKNOWN"

    close = df["close"]

    ma5 = close.rolling(5).mean().iloc[-1]
    ma10 = close.rolling(10).mean().iloc[-1]
    ma20 = close.rolling(20).mean().iloc[-1]

    price = close.iloc[-1]

    if price > ma5 > ma10 > ma20:
        return "BULLISH"

    if price < ma5 < ma10 < ma20:
        return "BEARISH"

    return "SIDEWAYS"


def safe_round(value, digits=2):
    if value is None:
        return None

    try:
        return round(float(value), digits)
    except:
        return None


@app.get("/")
def home():
    return {
        "status": "NIFTY AI backend running",
        "version": "V2.3"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "V2.3"
    }


@app.get("/nifty")
def nifty_analysis():

    try:

        # --------------------------------
        # CURRENT INTRADAY DATA
        # --------------------------------

        df, meta = fetch_yahoo("1d", "5m")

        if len(df) < 20:
            return {
                "error": "Not enough market data"
            }

        close = df["close"]

        price = float(close.iloc[-1])

        previous_close = meta.get("previousClose")

        if previous_close is None:
            previous_close = float(close.iloc[0])

        change = price - float(previous_close)

        change_percent = (
            change / float(previous_close) * 100
        )

        # --------------------------------
        # INDICATORS
        # --------------------------------

        ma5 = close.rolling(5).mean().iloc[-1]
        ma10 = close.rolling(10).mean().iloc[-1]
        ma20 = close.rolling(20).mean().iloc[-1]

        rsi14 = rsi(close, 14).iloc[-1]

        macd, macd_signal, macd_hist = calculate_macd(close)

        momentum = price - close.iloc[-6] if len(close) >= 6 else 0

        # --------------------------------
        # SUPPORT / RESISTANCE
        # --------------------------------

        recent = df.tail(min(30, len(df)))

        support = float(recent["low"].min())
        resistance = float(recent["high"].max())

        support_distance = (
            (price - support) / price * 100
        )

        resistance_distance = (
            (resistance - price) / price * 100
        )

        # --------------------------------
        # VOLUME
        # --------------------------------

        volume_status = "UNAVAILABLE"

        if "volume" in df.columns:

            volume = df["volume"].dropna()

            if len(volume) >= 10:

                current_volume = volume.iloc[-1]
                avg_volume = volume.tail(10).mean()

                if avg_volume > 0:

                    ratio = current_volume / avg_volume

                    if ratio >= 1.5:
                        volume_status = "HIGH"

                    elif ratio <= 0.7:
                        volume_status = "LOW"

                    else:
                        volume_status = "NORMAL"

        # --------------------------------
        # HIGHER TIMEFRAME
        # --------------------------------

        higher_tf_trend = "UNKNOWN"
        higher_tf_warning = None

        try:

            df_15m, _ = fetch_yahoo("5d", "15m")

            higher_tf_trend = trend_from_data(df_15m)

        except Exception:

            higher_tf_warning = "Higher timeframe data unavailable"

        # --------------------------------
        # SCORE ENGINE
        # --------------------------------

        bullish_score = 0
        bearish_score = 0

        reasons = []

        # Trend
        if price < ma5 and ma5 < ma10:
            bearish_score += 2
            reasons.append("Bearish short-term trend")

        elif price > ma5 and ma5 > ma10:
            bullish_score += 2
            reasons.append("Bullish short-term trend")

        # Higher timeframe
        if higher_tf_trend == "BEARISH":
            bearish_score += 2
            reasons.append("Higher timeframe bearish")

        elif higher_tf_trend == "BULLISH":
            bullish_score += 2
            reasons.append("Higher timeframe bullish")

        # RSI
        if rsi14 < 45:
            bearish_score += 1
            reasons.append("RSI bearish")

        elif rsi14 > 55:
            bullish_score += 1
            reasons.append("RSI bullish")

        # Momentum
        if momentum < 0:
            bearish_score += 1
            reasons.append("Negative momentum")

        elif momentum > 0:
            bullish_score += 1
            reasons.append("Positive momentum")

        # MACD
        if macd < macd_signal:
            bearish_score += 1
            reasons.append("MACD bearish")

        elif macd > macd_signal:
            bullish_score += 1
            reasons.append("MACD bullish")

        # --------------------------------
        # SUPPORT BREAKDOWN FILTER
        # --------------------------------

        near_support = support_distance <= 0.15

        breakdown_confirmed = False

        if price < support:

            breakdown_percent = (
                (support - price) / support * 100
            )

            if breakdown_percent >= 0.10:

                if (
                    momentum < 0
                    and macd < macd_signal
                ):
                    breakdown_confirmed = True

        # --------------------------------
        # RESISTANCE BREAKOUT FILTER
        # --------------------------------

        near_resistance = resistance_distance <= 0.15

        breakout_confirmed = False

        if price > resistance:

            breakout_percent = (
                (price - resistance) / resistance * 100
            )

            if breakout_percent >= 0.10:

                if (
                    momentum > 0
                    and macd > macd_signal
                ):
                    breakout_confirmed = True

        # --------------------------------
        # SIGNAL ENGINE V2.3
        # --------------------------------

        signal = "NEUTRAL"
        signal_strength = "LOW"

        warnings = []

        # IMPORTANT:
        # Don't aggressively SELL directly above support

        if near_support and not breakdown_confirmed:

            signal = "WAIT"
            signal_strength = "HIGH"

            warnings.append(
                "Price near support - wait for breakdown confirmation"
            )

        elif near_resistance and not breakout_confirmed:

            signal = "WAIT"
            signal_strength = "HIGH"

            warnings.append(
                "Price near resistance - wait for breakout confirmation"
            )

        else:

            if bullish_score >= 7:
                signal = "STRONG BUY"
                signal_strength = "HIGH"

            elif bullish_score >= 5:
                signal = "BUY"
                signal_strength = "MODERATE"

            elif bearish_score >= 7:

                if breakdown_confirmed:
                    signal = "STRONG SELL"
                    signal_strength = "HIGH"
                else:
                    signal = "SELL"
                    signal_strength = "MODERATE"

            elif bearish_score >= 5:

                if breakdown_confirmed:
                    signal = "SELL"
                    signal_strength = "MODERATE"
                else:
                    signal = "WAIT"
                    signal_strength = "HIGH"

            else:
                signal = "NEUTRAL"
                signal_strength = "LOW"

        # --------------------------------
        # CONFIDENCE
        # --------------------------------

        total_score = bullish_score + bearish_score

        if total_score > 0:

            confidence = (
                max(bullish_score, bearish_score)
                / total_score
            ) * 100

        else:
            confidence = 50

        # Avoid fake 90-100% confidence
        confidence = min(confidence, 85)

        # WAIT should not look like a high-confidence trade
        if signal == "WAIT":
            confidence = min(confidence, 60)

        # --------------------------------
        # ENTRY / AVOID ZONES
        # --------------------------------

        entry_zone = None
        avoid_zone = None

        if signal in ["BUY", "STRONG BUY"]:

            entry_zone = [
                safe_round(price),
                safe_round(resistance)
            ]

            avoid_zone = [
                safe_round(support),
                safe_round(support * 0.998)
            ]

        elif signal in ["SELL", "STRONG SELL"]:

            entry_zone = [
                safe_round(support * 0.999),
                safe_round(support * 0.995)
            ]

            avoid_zone = [
                safe_round(price),
                safe_round(resistance)
            ]

        elif signal == "WAIT":

            entry_zone = None

            avoid_zone = [
                safe_round(support),
                safe_round(resistance)
            ]

        # --------------------------------
        # MARKET STATUS
        # --------------------------------

        market_state = meta.get("marketState")

        if market_state == "REGULAR":
            market_status = "LIVE"

        elif market_state in ["PRE", "POST"]:
            market_status = market_state

        elif market_state == "CLOSED":
            market_status = "CLOSED"

        else:
            market_status = "UNKNOWN"

        # --------------------------------
        # VWAP
        # --------------------------------

        vwap = None

        # Yahoo index data may not provide usable volume
        if (
            "volume" in df.columns
            and df["volume"].notna().sum() > 0
            and df["volume"].sum() > 0
        ):

            typical_price = (
                df["high"] +
                df["low"] +
                df["close"]
            ) / 3

            cumulative_volume = df["volume"].cumsum()

            if cumulative_volume.iloc[-1] > 0:

                vwap = (
                    (typical_price * df["volume"]).cumsum()
                    / cumulative_volume
                ).iloc[-1]

        if vwap is None:
            warnings.append(
                "VWAP unavailable for NIFTY index"
            )

        if higher_tf_warning:
            warnings.append(higher_tf_warning)

        # --------------------------------
        # FINAL RESPONSE
        # --------------------------------

        return {

            "symbol": "NIFTY 50",

            "price": safe_round(price),

            "previous_close": safe_round(previous_close),

            "change": safe_round(change),

            "change_percent": safe_round(change_percent),

            "market_status": market_status,

            "trend": (
                "BULLISH"
                if bullish_score > bearish_score
                else "BEARISH"
                if bearish_score > bullish_score
                else "SIDEWAYS"
            ),

            "higher_timeframe_trend": higher_tf_trend,

            "signal": signal,

            "signal_strength": signal_strength,

            "confidence": safe_round(confidence),

            "rsi_14": safe_round(rsi14),

            "ma_5": safe_round(ma5),

            "ma_10": safe_round(ma10),

            "ma_20": safe_round(ma20),

            "vwap": safe_round(vwap),

            "macd": safe_round(macd),

            "macd_signal": safe_round(macd_signal),

            "macd_histogram": safe_round(macd_hist),

            "momentum": safe_round(momentum),

            "support": safe_round(support),

            "resistance": safe_round(resistance),

            "support_distance_percent":
                safe_round(support_distance, 3),

            "resistance_distance_percent":
                safe_round(resistance_distance, 3),

            "near_support": near_support,

            "near_resistance": near_resistance,

            "breakdown_confirmed":
                breakdown_confirmed,

            "breakout_confirmed":
                breakout_confirmed,

            "volume_status":
                volume_status,

            "bullish_score":
                bullish_score,

            "bearish_score":
                bearish_score,

            "reasons":
                reasons,

            "warnings":
                warnings,

            "entry_zone":
                entry_zone,

            "avoid_zone":
                avoid_zone,

            "analysis_version":
                "V2.3",

            "time":
                datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:

        return {
            "error": str(e),
            "analysis_version": "V2.3"
        }
