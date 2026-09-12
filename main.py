from fastapi import FastAPI
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone


APP_VERSION = "V2.6"

app = FastAPI(
    title="NIFTY AI Backend V2.6"
)


# =========================================================
# YAHOO DATA
# =========================================================

def fetch_yahoo(
    range_value="1d",
    interval="5m"
):

    url = (
        "https://query1.finance.yahoo.com/"
        "v8/finance/chart/%5ENSEI"
    )

    params = {
        "range": range_value,
        "interval": interval,
        "includePrePost": "false",
        "events": "div,splits"
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(
        url,
        params=params,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    chart = response.json()["chart"]

    result = chart["result"][0]

    timestamps = result.get(
        "timestamp",
        []
    )

    quote = result[
        "indicators"
    ]["quote"][0]

    df = pd.DataFrame({
        "time": pd.to_datetime(
            timestamps,
            unit="s"
        ),
        "open": quote.get(
            "open",
            []
        ),
        "high": quote.get(
            "high",
            []
        ),
        "low": quote.get(
            "low",
            []
        ),
        "close": quote.get(
            "close",
            []
        ),
        "volume": quote.get(
            "volume",
            []
        )
    })

    df = df.dropna(
        subset=["close"]
    ).reset_index(
        drop=True
    )

    return (
        df,
        result.get(
            "meta",
            {}
        )
    )


# =========================================================
# SAFE ROUND
# =========================================================

def safe_round(
    value,
    digits=2
):

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

        return round(
            float(value),
            digits
        )

    except Exception:

        return None


# =========================================================
# RSI
# =========================================================

def calculate_rsi(
    series,
    period=14
):

    delta = series.diff()

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    result = 100 - (
        100 /
        (1 + rs)
    )

    return result


# =========================================================
# MACD
# =========================================================

def calculate_macd(
    series
):

    ema12 = series.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = series.ewm(
        span=26,
        adjust=False
    ).mean()

    macd = (
        ema12 -
        ema26
    )

    signal = macd.ewm(
        span=9,
        adjust=False
    ).mean()

    histogram = (
        macd -
        signal
    )

    return (
        macd.iloc[-1],
        signal.iloc[-1],
        histogram.iloc[-1]
    )


# =========================================================
# TREND
# =========================================================

def trend_from_data(
    df
):

    if len(df) < 20:
        return "UNKNOWN"

    close = df["close"]

    ma5 = close.rolling(
        5
    ).mean().iloc[-1]

    ma10 = close.rolling(
        10
    ).mean().iloc[-1]

    ma20 = close.rolling(
        20
    ).mean().iloc[-1]

    price = close.iloc[-1]

    if (
        price >
        ma5 >
        ma10 >
        ma20
    ):
        return "BULLISH"

    if (
        price <
        ma5 <
        ma10 <
        ma20
    ):
        return "BEARISH"

    return "SIDEWAYS"


# =========================================================
# VOLUME
# =========================================================

def calculate_volume(
    df
):

    volume_status = "UNAVAILABLE"

    volume_ratio = None

    if "volume" not in df.columns:
        return (
            volume_status,
            volume_ratio
        )

    volume = df[
        "volume"
    ].dropna()

    volume = volume[
        volume > 0
    ]

    if len(volume) < 10:
        return (
            volume_status,
            volume_ratio
        )

    current_volume = float(
        volume.iloc[-1]
    )

    average_volume = float(
        volume.tail(10).mean()
    )

    if average_volume <= 0:
        return (
            volume_status,
            volume_ratio
        )

    volume_ratio = (
        current_volume /
        average_volume
    )

    if volume_ratio >= 1.5:

        volume_status = "HIGH"

    elif volume_ratio <= 0.7:

        volume_status = "LOW"

    else:

        volume_status = "NORMAL"

    return (
        volume_status,
        volume_ratio
    )


# =========================================================
# VWAP
# =========================================================

def calculate_vwap(
    df
):

    if "volume" not in df.columns:
        return None

    if (
        df["volume"]
        .notna()
        .sum()
        == 0
    ):
        return None

    if (
        df["volume"]
        .sum()
        <= 0
    ):
        return None

    typical_price = (
        df["high"] +
        df["low"] +
        df["close"]
    ) / 3

    cumulative_volume = (
        df["volume"]
        .cumsum()
    )

    if (
        cumulative_volume.iloc[-1]
        <= 0
    ):
        return None

    vwap = (
        (
            typical_price *
            df["volume"]
        ).cumsum()
        /
        cumulative_volume
    ).iloc[-1]

    return vwap


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "status":
            "NIFTY AI backend running",

        "version":
            APP_VERSION
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",

        "version":
            APP_VERSION
    }


# =========================================================
# NIFTY ANALYSIS
# =========================================================

@app.get("/nifty")
def nifty_analysis():

    try:

        # =================================================
        # 5 MINUTE DATA
        # =================================================

        df, meta = fetch_yahoo(
            "1d",
            "5m"
        )

        if len(df) < 30:

            return {
                "error":
                    "Not enough market data",

                "analysis_version":
                    APP_VERSION
            }


        close = df["close"]

        price = float(
            close.iloc[-1]
        )


        # =================================================
        # PREVIOUS CLOSE
        # =================================================

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


        # =================================================
        # MOVING AVERAGES
        # =================================================

        ma5 = close.rolling(
            5
        ).mean().iloc[-1]

        ma10 = close.rolling(
            10
        ).mean().iloc[-1]

        ma20 = close.rolling(
            20
        ).mean().iloc[-1]


        # =================================================
        # RSI
        # =================================================

        rsi_series = calculate_rsi(
            close,
            14
        )

        rsi14 = rsi_series.iloc[-1]


        # =================================================
        # MACD
        # =================================================

        (
            macd,
            macd_signal,
            macd_hist
        ) = calculate_macd(
            close
        )


        # =================================================
        # MOMENTUM
        # =================================================

        if len(df) >= 6:

            momentum = (
                price -
                float(
                    close.iloc[-6]
                )
            )

        else:

            momentum = 0.0


        # =================================================
        # SUPPORT / RESISTANCE
        # =================================================

        completed = df.iloc[:-1]

        recent = completed.tail(
            min(
                30,
                len(completed)
            )
        )

        support = float(
            recent["low"].min()
        )

        resistance = float(
            recent["high"].max()
        )


        support_distance = (
            (price - support) /
            price *
            100
        )

        resistance_distance = (
            (resistance - price) /
            price *
            100
        )


        # =================================================
        # NEAR LEVELS
        # =================================================

        near_support = (
            support_distance <= 0.20
        )

        near_resistance = (
            resistance_distance <= 0.20
        )


        # =================================================
        # VOLUME
        # =================================================

        (
            volume_status,
            volume_ratio
        ) = calculate_volume(
            df
        )


        # =================================================
        # HIGHER TIMEFRAME
        # =================================================

        higher_tf_trend = "UNKNOWN"

        higher_tf_warning = None

        try:

            df_15m, _ = fetch_yahoo(
                "5d",
                "15m"
            )

            higher_tf_trend = (
                trend_from_data(
                    df_15m
                )
            )

        except Exception:

            higher_tf_warning = (
                "Higher timeframe data unavailable"
            )


        # =================================================
        # BREAKOUT / BREAKDOWN
        # =================================================

        breakout_confirmed = False

        breakdown_confirmed = False

        false_breakout = False

        false_breakdown = False


        if len(df) >= 3:

            candle1 = df.iloc[-3]

            candle2 = df.iloc[-2]

        else:

            candle1 = None

            candle2 = None


        # =================================================
        # BREAKOUT CHECK
        # =================================================

        if candle2 is not None:

            breakout_percent = (
                (price - resistance) /
                resistance *
                100
            )

            if (
                price > resistance
                and
                breakout_percent >= 0.10
            ):

                candle_confirmation = (
                    float(
                        candle2["close"]
                    )
                    >
                    resistance
                )

                momentum_confirmation = (
                    momentum > 0
                )

                macd_confirmation = (
                    macd > macd_signal
                    and
                    macd_hist > 0
                )

                volume_confirmation = True

                if (
                    volume_status !=
                    "UNAVAILABLE"
                ):

                    volume_confirmation = (
                        volume_ratio is not None
                        and
                        volume_ratio >= 1.05
                    )

                if (
                    candle_confirmation
                    and
                    momentum_confirmation
                    and
                    macd_confirmation
                    and
                    volume_confirmation
                ):

                    breakout_confirmed = True

                elif (
                    candle_confirmation
                    and
                    momentum_confirmation
                    and
                    macd_confirmation
                    and
                    volume_status ==
                    "UNAVAILABLE"
                ):

                    breakout_confirmed = True

                else:

                    false_breakout = True


        # =================================================
        # BREAKDOWN CHECK
        # =================================================

        if candle2 is not None:

            breakdown_percent = (
                (support - price) /
                support *
                100
            )

            if (
                price < support
                and
                breakdown_percent >= 0.10
            ):

                candle_confirmation = (
                    float(
                        candle2["close"]
                    )
                    <
                    support
                )

                momentum_confirmation = (
                    momentum < 0
                )

                macd_confirmation = (
                    macd < macd_signal
                    and
                    macd_hist < 0
                )

                volume_confirmation = True

                if (
                    volume_status !=
                    "UNAVAILABLE"
                ):

                    volume_confirmation = (
                        volume_ratio is not None
                        and
                        volume_ratio >= 1.05
                    )

                if (
                    candle_confirmation
                    and
                    momentum_confirmation
                    and
                    macd_confirmation
                    and
                    volume_confirmation
                ):

                    breakdown_confirmed = True

                elif (
                    candle_confirmation
                    and
                    momentum_confirmation
                    and
                    macd_confirmation
                    and
                    volume_status ==
                    "UNAVAILABLE"
                ):

                    breakdown_confirmed = True

                else:

                    false_breakdown = True


        # =================================================
        # SCORE ENGINE V2.6
        # =================================================

        bullish_score = 0

        bearish_score = 0

        reasons = []


        # =================================================
        # MA ALIGNMENT
        # =================================================

        bullish_ma = (
            price > ma5
            and
            ma5 > ma10
            and
            ma10 > ma20
        )

        bearish_ma = (
            price < ma5
            and
            ma5 < ma10
            and
            ma10 < ma20
        )


        if bullish_ma:

            bullish_score += 3

            reasons.append(
                "Bullish MA alignment"
            )

        elif bearish_ma:

            bearish_score += 3

            reasons.append(
                "Bearish MA alignment"
            )


        # =================================================
        # HIGHER TIMEFRAME
        # =================================================

        if (
            higher_tf_trend ==
            "BULLISH"
        ):

            bullish_score += 2

            reasons.append(
                "Higher timeframe bullish"
            )

        elif (
            higher_tf_trend ==
            "BEARISH"
        ):

            bearish_score += 2

            reasons.append(
                "Higher timeframe bearish"
            )


        # =================================================
        # RSI
        # =================================================

        if rsi14 >= 55:

            bullish_score += 2

            reasons.append(
                "RSI bullish"
            )

        elif rsi14 <= 45:

            bearish_score += 2

            reasons.append(
                "RSI bearish"
            )


        # =================================================
        # MACD
        # =================================================

        if (
            macd > macd_signal
            and
            macd_hist > 0
        ):

            bullish_score += 2

            reasons.append(
                "MACD bullish"
            )

        elif (
            macd < macd_signal
            and
            macd_hist < 0
        ):

            bearish_score += 2

            reasons.append(
                "MACD bearish"
            )


        # =================================================
        # MOMENTUM
        # =================================================

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


        # =================================================
        # VWAP
        # =================================================

        vwap = calculate_vwap(
            df
        )

        if vwap is not None:

            if price > vwap:

                bullish_score += 1

                reasons.append(
                    "Price above VWAP"
                )

            elif price < vwap:

                bearish_score += 1

                reasons.append(
                    "Price below VWAP"
                )


        # =================================================
        # BREAKOUT / BREAKDOWN SCORE
        # =================================================

        if breakout_confirmed:

            bullish_score += 3

            reasons.append(
                "Confirmed resistance breakout"
            )

        if breakdown_confirmed:

            bearish_score += 3

            reasons.append(
                "Confirmed support breakdown"
            )


        # =================================================
        # HIGHER TIMEFRAME CONFLICT
        # =================================================

        higher_tf_conflict = False

        if (
            bullish_score >= 6
            and
            higher_tf_trend ==
            "BEARISH"
        ):

            higher_tf_conflict = True

        if (
            bearish_score >= 6
            and
            higher_tf_trend ==
            "BULLISH"
        ):

            higher_tf_conflict = True


        # =================================================
        # SIGNAL ENGINE
        # =================================================

        signal = "NEUTRAL"

        signal_strength = "LOW"


        # =================================================
        # STRONG SELL
        # =================================================

        if breakdown_confirmed:

            signal = "STRONG SELL"

            signal_strength = "HIGH"


        # =================================================
        # STRONG BUY
        # =================================================

        elif breakout_confirmed:

            signal = "STRONG BUY"

            signal_strength = "HIGH"


        # =================================================
        # FALSE BREAK
        # =================================================

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


        # =================================================
        # LEVEL PROTECTION
        # =================================================

        elif near_support:

            signal = "WAIT"

            signal_strength = "HIGH"

            reasons.append(
                "Price near support"
            )

        elif near_resistance:

            signal = "WAIT"

            signal_strength = "HIGH"

            reasons.append(
                "Price near resistance"
            )


        # =================================================
        # CONFLICT PROTECTION
        # =================================================

        elif higher_tf_conflict:

            signal = "WAIT"

            signal_strength = "HIGH"

            reasons.append(
                "Higher timeframe conflict"
            )


        # =================================================
        # NORMAL BUY / SELL
        # =================================================

        elif (
            bullish_score >= 7
            and
            bullish_score >
            bearish_score + 2
        ):

            signal = "BUY"

            signal_strength = "MODERATE"

        elif (
            bearish_score >= 7
            and
            bearish_score >
            bullish_score + 2
        ):

            signal = "SELL"

            signal_strength = "MODERATE"

        else:

            signal = "NEUTRAL"

            signal_strength = "LOW"


        # =================================================
        # CONFIDENCE
        # =================================================

        total_score = (
            bullish_score +
            bearish_score
        )

        if total_score > 0:

            dominant_score = max(
                bullish_score,
                bearish_score
            )

            confidence = (
                dominant_score /
                total_score
            ) * 100

        else:

            confidence = 50.0


        # =================================================
        # CONFIDENCE CAPS
        # =================================================

        if signal == "NEUTRAL":

            confidence = min(
                confidence,
                60
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
                85
            )

        elif signal in [
            "STRONG BUY",
            "STRONG SELL"
        ]:

            confidence = min(
                confidence,
                92
            )


        # =================================================
        # ENTRY / AVOID ZONES
        # =================================================

        entry_zone = None

        avoid_zone = None


        if signal == "STRONG BUY":

            entry_zone = [
                safe_round(
                    resistance
                ),
                safe_round(
                    resistance *
                    1.002
                )
            ]

            avoid_zone = [
                safe_round(
                    support
                ),
                safe_round(
                    support *
                    0.998
                )
            ]


        elif signal == "BUY":

            entry_zone = [
                safe_round(
                    price
                ),
                safe_round(
                    resistance
                )
            ]

            avoid_zone = [
                safe_round(
                    support
                ),
                safe_round(
                    support *
                    0.998
                )
            ]


        elif signal == "STRONG SELL":

            entry_zone = [
                safe_round(
                    support *
                    0.998
                ),
                safe_round(
                    support *
                    0.995
                )
            ]

            avoid_zone = [
                safe_round(
                    price
                ),
                safe_round(
                    resistance
                )
            ]


        elif signal == "SELL":

            entry_zone = [
                safe_round(
                    support *
                    0.999
                ),
                safe_round(
                    support *
                    0.995
                )
            ]

            avoid_zone = [
                safe_round(
                    price
                ),
                safe_round(
                    resistance
                )
            ]


        elif signal == "WAIT":

            entry_zone = None

            avoid_zone = [
                safe_round(
                    support
                ),
                safe_round(
                    resistance
                )
            ]


        # =================================================
        # MARKET STATUS
        # =================================================

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


        # =================================================
        # WARNINGS
        # =================================================

        warnings = []


        if vwap is None:

            warnings.append(
                "VWAP unavailable for NIFTY index"
            )


        if (
            volume_status ==
            "UNAVAILABLE"
        ):

            warnings.append(
                "Volume data unavailable"
            )


        if higher_tf_warning:

            warnings.append(
                higher_tf_warning
            )


        if higher_tf_conflict:

            warnings.append(
                "Short-term and higher-timeframe "
                "signals are conflicting"
            )


        # =================================================
        # FINAL TREND
        # =================================================

        if (
            bullish_score >
            bearish_score
        ):

            final_trend = "BULLISH"

        elif (
            bearish_score >
            bullish_score
        ):

            final_trend = "BEARISH"

        else:

            final_trend = "SIDEWAYS"


        # =================================================
        # FINAL RESPONSE
        # =================================================

        return {

            "symbol":
                "NIFTY 50",

            "price":
                safe_round(price),

            "previous_close":
                safe_round(
                    previous_close
                ),

            "change":
                safe_round(
                    change
                ),

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
                safe_round(
                    confidence
                ),

            "rsi_14":
                safe_round(
                    rsi14
                ),

            "ma_5":
                safe_round(
                    ma5
                ),

            "ma_10":
                safe_round(
                    ma10
                ),

            "ma_20":
                safe_round(
                    ma20
                ),

            "vwap":
                safe_round(
                    vwap
                ),

            "macd":
                safe_round(
                    macd
                ),

            "macd_signal":
                safe_round(
                    macd_signal
                ),

            "macd_histogram":
                safe_round(
                    macd_hist
                ),

            "momentum":
                safe_round(
                    momentum
                ),

            "support":
                safe_round(
                    support
                ),

            "resistance":
                safe_round(
                    resistance
                ),

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

            "higher_tf_conflict":
                higher_tf_conflict,

            "reasons":
                reasons,

            "warnings":
                warnings,

            "entry_zone":
                entry_zone,

            "avoid_zone":
                avoid_zone,

            "analysis_version":
                APP_VERSION,

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
                APP_VERSION
        }
