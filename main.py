from fastapi import FastAPI
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

app = FastAPI(title="NIFTY AI Backend V2.4")


# =========================
# YAHOO DATA
# =========================

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

    r = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=15
    )

    r.raise_for_status()

    result = r.json()["chart"]["result"][0]

    timestamps = result.get("timestamp", [])

    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "time": pd.to_datetime(
            timestamps,
            unit="s"
        ),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", [])
    })

    df = df.dropna(
        subset=["close"]
    ).reset_index(drop=True)

    return df, result.get("meta", {})


# =========================
# RSI
# =========================

def rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()

    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    return 100 - (
        100 / (1 + rs)
    )


# =========================
# MACD
# =========================

def calculate_macd(series):

    ema12 = series.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = series.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = ema12 - ema26

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = macd - signal

    return (
        macd.iloc[-1],
        signal.iloc[-1],
        histogram.iloc[-1]
    )


# =========================
# TREND
# =========================

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


# =========================
# ROUNDING
# =========================

def safe_round(value, digits=2):

    if value is None:
        return None

    try:
        return round(float(value), digits)

    except:
        return None


# =========================
# HOME
# =========================

@app.get("/")
def home():

    return {
        "status": "NIFTY AI backend running",
        "version": "V2.4"
    }


# =========================
# HEALTH
# =========================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "version": "V2.4"
    }


# =========================
# NIFTY ANALYSIS
# =========================

@app.get("/nifty")
def nifty_analysis():

    try:

        # --------------------------------
        # CURRENT 5 MINUTE DATA
        # --------------------------------

        df, meta = fetch_yahoo(
            "1d",
            "5m"
        )

        if len(df) < 30:

            return {
                "error": "Not enough market data",
                "analysis_version": "V2.4"
            }


        close = df["close"]

        price = float(
            close.iloc[-1]
        )


        # --------------------------------
        # PREVIOUS CLOSE
        # --------------------------------

        previous_close = meta.get(
            "previousClose"
        )

        if previous_close is None:

            previous_close = float(
                close.iloc[0]
            )

        previous_close = float(
            previous_close
        )

        change = (
            price -
            previous_close
        )

        change_percent = (
            change /
            previous_close *
            100
        )


        # --------------------------------
        # MOVING AVERAGES
        # --------------------------------

        ma5 = close.rolling(
            5
        ).mean().iloc[-1]

        ma10 = close.rolling(
            10
        ).mean().iloc[-1]

        ma20 = close.rolling(
            20
        ).mean().iloc[-1]


        # --------------------------------
        # RSI
        # --------------------------------

        rsi14 = rsi(
            close,
            14
        ).iloc[-1]


        # --------------------------------
        # MACD
        # --------------------------------

        macd, macd_signal, macd_hist = (
            calculate_macd(close)
        )


        # --------------------------------
        # MOMENTUM
        # --------------------------------

        if len(df) >= 6:

            momentum = (
                price -
                float(close.iloc[-6])
            )

        else:

            momentum = 0


        # =================================
        # SUPPORT / RESISTANCE
        # =================================

        # Current candle is excluded.
        # This prevents the current candle
        # from artificially changing levels.

        completed = df.iloc[:-1]

        recent = completed.tail(
            min(30, len(completed))
        )

        support = float(
            recent["low"].min()
        )

        resistance = float(
            recent["high"].max()
        )


        support_distance = (
            (price - support)
            / price *
            100
        )

        resistance_distance = (
            (resistance - price)
            / price *
            100
        )


        # =================================
        # NEAR SUPPORT / RESISTANCE
        # =================================

        near_support = (
            support_distance <= 0.15
        )

        near_resistance = (
            resistance_distance <= 0.15
        )


        # =================================
        # BREAKOUT / BREAKDOWN
        # =================================

        breakdown_confirmed = False

        breakout_confirmed = False

        false_breakdown = False

        false_breakout = False


        # --------------------------------
        # PREVIOUS TWO COMPLETED CANDLES
        # --------------------------------

        if len(df) >= 3:

            candle1 = df.iloc[-3]

            candle2 = df.iloc[-2]

        else:

            candle1 = None

            candle2 = None


        # --------------------------------
        # BREAKDOWN
        # --------------------------------

        if candle2 is not None:

            breakdown_percent = (
                (support - price)
                / support *
                100
            )

            if price < support:

                if breakdown_percent >= 0.10:

                    candle_confirmation = (
                        float(candle2["close"])
                        < support
                    )

                    momentum_confirmation = (
                        momentum < 0
                    )

                    macd_confirmation = (
                        macd < macd_signal
                    )

                    volume_confirmation = True


                    # Volume confirmation only
                    # when usable volume exists.

                    volume = df["volume"].dropna()

                    if len(volume) >= 10:

                        current_volume = (
                            volume.iloc[-1]
                        )

                        avg_volume = (
                            volume.tail(10)
                            .mean()
                        )

                        if avg_volume > 0:

                            volume_ratio = (
                                current_volume /
                                avg_volume
                            )

                            volume_confirmation = (
                                volume_ratio >= 1.10
                            )


                    if (
                        candle_confirmation
                        and momentum_confirmation
                        and macd_confirmation
                        and volume_confirmation
                    ):

                        breakdown_confirmed = True

                    elif (
                        candle_confirmation
                        and momentum_confirmation
                        and macd_confirmation
                    ):

                        # Allow confirmation without
                        # volume because NIFTY index
                        # volume may be unavailable.

                        breakdown_confirmed = True


                elif price < support:

                    false_breakdown = True


        # --------------------------------
        # BREAKOUT
        # --------------------------------

        if candle2 is not None:

            breakout_percent = (
                (price - resistance)
                / resistance *
                100
            )

            if price > resistance:

                if breakout_percent >= 0.10:

                    candle_confirmation = (
                        float(candle2["close"])
                        > resistance
                    )

                    momentum_confirmation = (
                        momentum > 0
                    )

                    macd_confirmation = (
                        macd > macd_signal
                    )

                    volume_confirmation = True


                    volume = df["volume"].dropna()

                    if len(volume) >= 10:

                        current_volume = (
                            volume.iloc[-1]
                        )

                        avg_volume = (
                            volume.tail(10)
                            .mean()
                        )

                        if avg_volume > 0:

                            volume_ratio = (
                                current_volume /
                                avg_volume
                            )

                            volume_confirmation = (
                                volume_ratio >= 1.10
                            )


                    if (
                        candle_confirmation
                        and momentum_confirmation
                        and macd_confirmation
                        and volume_confirmation
                    ):

                        breakout_confirmed = True

                    elif (
                        candle_confirmation
                        and momentum_confirmation
                        and macd_confirmation
                    ):

                        breakout_confirmed = True


                elif price > resistance:

                    false_breakout = True


        # =================================
        # VOLUME
        # =================================

        volume_status = "UNAVAILABLE"

        volume_ratio = None

        volume = df["volume"].dropna()

        if len(volume) >= 10:

            current_volume = volume.iloc[-1]

            avg_volume = (
                volume.tail(10)
                .mean()
            )

            if avg_volume > 0:

                volume_ratio = (
                    current_volume /
                    avg_volume
                )

                if volume_ratio >= 1.5:

                    volume_status = "HIGH"

                elif volume_ratio <= 0.7:

                    volume_status = "LOW"

                else:

                    volume_status = "NORMAL"


        # =================================
        # HIGHER TIMEFRAME
        # =================================

        higher_tf_trend = "UNKNOWN"

        higher_tf_warning = None

        try:

            df_15m, _ = fetch_yahoo(
                "5d",
                "15m"
            )

            higher_tf_trend = (
                trend_from_data(df_15m)
            )

        except Exception:

            higher_tf_warning = (
                "Higher timeframe data unavailable"
            )


        # =================================
        # SCORE ENGINE
        # =================================

        bullish_score = 0

        bearish_score = 0

        reasons = []


        # --------------------------------
        # SHORT TERM TREND
        # --------------------------------

        if price > ma5 > ma10:

            bullish_score += 2

            reasons.append(
                "Bullish short-term trend"
            )

        elif price < ma5 < ma10:

            bearish_score += 2

            reasons.append(
                "Bearish short-term trend"
            )


        # --------------------------------
        # HIGHER TIMEFRAME
        # --------------------------------

        if higher_tf_trend == "BULLISH":

            bullish_score += 2

            reasons.append(
                "Higher timeframe bullish"
            )

        elif higher_tf_trend == "BEARISH":

            bearish_score += 2

            reasons.append(
                "Higher timeframe bearish"
            )


        # --------------------------------
        # RSI
        # --------------------------------

        if rsi14 > 55:

            bullish_score += 1

            reasons.append(
                "RSI bullish"
            )

        elif rsi14 < 45:

            bearish_score += 1

            reasons.append(
                "RSI bearish"
            )


        # --------------------------------
        # MOMENTUM
        # --------------------------------

        if momentum > 0:

            bullish_score += 1

            reasons.append(
                "Positive momentum"
            )

        elif momentum < 0:

            bearish_score += 1

            reasons.append(
                "Negative momentum"
            )


        # --------------------------------
        # MACD
        # --------------------------------

        if macd > macd_signal:

            bullish_score += 1

            reasons.append(
                "MACD bullish"
            )

        elif macd < macd_signal:

            bearish_score += 1

            reasons.append(
                "MACD bearish"
            )


        # =================================
        # SIGNAL ENGINE
        # =================================

        signal = "NEUTRAL"

        signal_strength = "LOW"


        # --------------------------------
        # CONFIRMED BREAKDOWN
        # --------------------------------

        if breakdown_confirmed:

            signal = "STRONG SELL"

            signal_strength = "HIGH"

            reasons.append(
                "Confirmed support breakdown"
            )


        # --------------------------------
        # CONFIRMED BREAKOUT
        # --------------------------------

        elif breakout_confirmed:

            signal = "STRONG BUY"

            signal_strength = "HIGH"

            reasons.append(
                "Confirmed resistance breakout"
            )


        # --------------------------------
        # FALSE BREAKOUT / BREAKDOWN
        # --------------------------------

        elif false_breakdown:

            signal = "WAIT"

            signal_strength = "HIGH"

            reasons.append(
                "Possible false breakdown"
            )


        elif false_breakout:

            signal = "WAIT"

            signal_strength = "HIGH"

            reasons.append(
                "Possible false breakout"
            )


        # --------------------------------
        # NEAR SUPPORT
        # --------------------------------

        elif near_support:

            signal = "WAIT"

            signal_strength = "HIGH"

            reasons.append(
                "Price near support - "
                "wait for breakdown confirmation"
            )


        # --------------------------------
        # NEAR RESISTANCE
        # --------------------------------

        elif near_resistance:

            signal = "WAIT"

            signal_strength = "HIGH"

            reasons.append(
                "Price near resistance - "
                "wait for breakout confirmation"
            )


        # --------------------------------
        # NORMAL SCORE SIGNAL
        # --------------------------------

        else:

            if bullish_score >= 6:

                signal = "BUY"

                signal_strength = "MODERATE"


            elif bearish_score >= 6:

                signal = "SELL"

                signal_strength = "MODERATE"


            else:

                signal = "NEUTRAL"

                signal_strength = "LOW"


        # =================================
        # CONFIDENCE
        # =================================

        total_score = (
            bullish_score +
            bearish_score
        )

        if total_score > 0:

            confidence = (
                max(
                    bullish_score,
                    bearish_score
                )
                /
                total_score
            ) * 100

        else:

            confidence = 50


        # Avoid misleadingly high
        # confidence for weak signals.

        if signal == "NEUTRAL":

            confidence = min(
                confidence,
                65
            )

        elif signal == "WAIT":

            confidence = min(
                confidence,
                70
            )

        elif signal in [
            "BUY",
            "SELL"
        ]:

            confidence = min(
                confidence,
                80
            )

        else:

            confidence = min(
                confidence,
                90
            )


        # =================================
        # ENTRY / AVOID ZONES
        # =================================

        entry_zone = None

        avoid_zone = None


        if signal == "STRONG BUY":

            entry_zone = [
                safe_round(resistance),
                safe_round(
                    resistance * 1.002
                )
            ]

            avoid_zone = [
                safe_round(support),
                safe_round(
                    support * 0.998
                )
            ]


        elif signal == "BUY":

            entry_zone = [
                safe_round(price),
                safe_round(resistance)
            ]

            avoid_zone = [
                safe_round(support),
                safe_round(
                    support * 0.998
                )
            ]


        elif signal == "STRONG SELL":

            entry_zone = [
                safe_round(
                    support * 0.998
                ),
                safe_round(
                    support * 0.995
                )
            ]

            avoid_zone = [
                safe_round(price),
                safe_round(resistance)
            ]


        elif signal == "SELL":

            entry_zone = [
                safe_round(
                    support * 0.999
                ),
                safe_round(
                    support * 0.995
                )
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


        # =================================
        # MARKET STATUS
        # =================================

        market_state = meta.get(
            "marketState"
        )

        if market_state == "REGULAR":

            market_status = "LIVE"

        elif market_state == "PRE":

            market_status = "PRE"

        elif market_state == "POST":

            market_status = "POST"

        elif market_state == "CLOSED":

            market_status = "CLOSED"

        else:

            market_status = "UNKNOWN"


        # =================================
        # VWAP
        # =================================

        vwap = None

        if (
            "volume" in df.columns
            and
            df["volume"].notna().sum() > 0
            and
            df["volume"].sum() > 0
        ):

            typical_price = (
                df["high"] +
                df["low"] +
                df["close"]
            ) / 3

            cumulative_volume = (
                df["volume"].cumsum()
            )

            if (
                cumulative_volume.iloc[-1]
                > 0
            ):

                vwap = (
                    (
                        typical_price *
                        df["volume"]
                    ).cumsum()
                    /
                    cumulative_volume
                ).iloc[-1]


        # =================================
        # WARNINGS
        # =================================

        warnings = []

        if vwap is None:

            warnings.append(
                "VWAP unavailable for NIFTY index"
            )

        if volume_status == "UNAVAILABLE":

            warnings.append(
                "Volume data unavailable"
            )

        if higher_tf_warning:

            warnings.append(
                higher_tf_warning
            )


        # =================================
        # FINAL TREND
        # =================================

        if bullish_score > bearish_score:

            final_trend = "BULLISH"

        elif bearish_score > bullish_score:

            final_trend = "BEARISH"

        else:

            final_trend = "SIDEWAYS"


        # =================================
        # FINAL RESPONSE
        # =================================

        return {

            "symbol":
                "NIFTY 50",

            "price":
                safe_round(price),

            "previous_close":
                safe_round(previous_close),

            "change":
                safe_round(change),

            "change_percent":
                safe_round(
                    change_percent
                ),

            "market_status":
                market_status,

            "trend":
                final_trend,

            "higher_timeframe_trend":
                higher_tf_trend,

            "signal":
                signal,

            "signal_strength":
                signal_strength,

            "confidence":
                safe_round(confidence),

            "rsi_14":
                safe_round(rsi14),

            "ma_5":
                safe_round(ma5),

            "ma_10":
                safe_round(ma10),

            "ma_20":
                safe_round(ma20),

            "vwap":
                safe_round(vwap),

            "macd":
                safe_round(macd),

            "macd_signal":
                safe_round(macd_signal),

            "macd_histogram":
                safe_round(macd_hist),

            "momentum":
                safe_round(momentum),

            "support":
                safe_round(support),

            "resistance":
                safe_round(resistance),

            "support_distance_percent":
                safe_round(
                    support_distance,
                    3
                ),

            "resistance_distance_percent":
                safe_round(
                    resistance_distance,
                    3
                ),

            "near_support":
                near_support,

            "near_resistance":
                near_resistance,

            "breakdown_confirmed":
                breakdown_confirmed,

            "breakout_confirmed":
                breakout_confirmed,

            "false_breakdown":
                false_breakdown,

            "false_breakout":
                false_breakout,

            "volume_status":
                volume_status,

            "volume_ratio":
                safe_round(
                    volume_ratio,
                    2
                ),

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
                "V2.4",

            "time":
                datetime.now(
                    timezone.utc
                ).isoformat()
        }


    except Exception as e:

        return {

            "error":
                str(e),

            "analysis_version":
                "V2.4"
        }
