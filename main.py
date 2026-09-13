from fastapi import FastAPI, Query
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

app = FastAPI(title="NIFTY AI Backend V3.1")


# =========================================================
# CONFIG
# =========================================================

SYMBOL = "^NSEI"
YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/%5ENSEI"

ANALYSIS_VERSION = "V3.1"

NEAR_LEVEL_PERCENT = 0.20
BREAK_CONFIRM_PERCENT = 0.10

MIN_SCORE = 7
MIN_SCORE_GAP = 2

ATR_PERIOD = 14

DEFAULT_RISK_PERCENT = 0.60
MIN_RR = 1.50


# =========================================================
# YAHOO DATA
# =========================================================

def fetch_yahoo(range_value="1d", interval="5m"):

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
        YAHOO_URL,
        params=params,
        headers=headers,
        timeout=15
    )

    response.raise_for_status()

    data = response.json()

    if "chart" not in data:
        raise Exception("Yahoo response missing chart")

    if not data["chart"].get("result"):
        raise Exception("No Yahoo market data available")

    result = data["chart"]["result"][0]

    timestamps = result.get("timestamp", [])

    quote = result["indicators"]["quote"][0]

    df = pd.DataFrame({
        "time": pd.to_datetime(
            timestamps,
            unit="s",
            utc=True
        ),
        "open": quote.get("open", []),
        "high": quote.get("high", []),
        "low": quote.get("low", []),
        "close": quote.get("close", []),
        "volume": quote.get("volume", [])
    })

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    ).reset_index(drop=True)

    return df, result.get("meta", {})


# =========================================================
# SAFE ROUND
# =========================================================

def safe_round(value, digits=2):

    if value is None:
        return None

    try:

        if pd.isna(value):
            return None

        return round(float(value), digits)

    except Exception:

        return None


# =========================================================
# RSI
# =========================================================

def calculate_rsi(series, period=14):

    delta = series.diff()

    gain = delta.clip(lower=0)

    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(
        period
    ).mean()

    avg_loss = loss.rolling(
        period
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(0, np.nan)
    )

    return 100 - (
        100 / (1 + rs)
    )


# =========================================================
# MACD
# =========================================================

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
        float(macd.iloc[-1]),
        float(signal.iloc[-1]),
        float(histogram.iloc[-1])
    )


# =========================================================
# ATR
# =========================================================

def calculate_atr(df, period=14):

    previous_close = df["close"].shift(1)

    tr1 = df["high"] - df["low"]

    tr2 = (
        df["high"] -
        previous_close
    ).abs()

    tr3 = (
        df["low"] -
        previous_close
    ).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    atr = true_range.rolling(
        period
    ).mean()

    value = atr.iloc[-1]

    if pd.isna(value):

        return float(
            (df["high"] - df["low"])
            .tail(period)
            .mean()
        )

    return float(value)


# =========================================================
# TREND
# =========================================================

def calculate_trend(df):

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

    if price > ma5 > ma10 > ma20:
        return "BULLISH"

    if price < ma5 < ma10 < ma20:
        return "BEARISH"

    return "SIDEWAYS"


# =========================================================
# VOLUME
# =========================================================

def calculate_volume(df):

    volume = df["volume"].dropna()

    if len(volume) < 10:
        return "UNAVAILABLE", None

    current = float(
        volume.iloc[-1]
    )

    average = float(
        volume.tail(10).mean()
    )

    if average <= 0:
        return "UNAVAILABLE", None

    ratio = current / average

    if ratio >= 1.50:

        status = "HIGH"

    elif ratio <= 0.70:

        status = "LOW"

    else:

        status = "NORMAL"

    return status, ratio


# =========================================================
# SUPPORT / RESISTANCE
# =========================================================

def calculate_levels(df):

    if len(df) < 10:
        return None, None

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

    return support, resistance


# =========================================================
# BREAKOUT / BREAKDOWN
# =========================================================

def detect_breakout_breakdown(
    df,
    support,
    resistance,
    momentum,
    macd,
    macd_signal
):

    breakout = False
    breakdown = False

    false_breakout = False
    false_breakdown = False

    if len(df) < 3:

        return (
            breakout,
            breakdown,
            false_breakout,
            false_breakdown
        )

    price = float(
        df["close"].iloc[-1]
    )

    previous = df.iloc[-2]

    # =====================================================
    # BREAKOUT
    # =====================================================

    if price > resistance:

        distance = (
            (price - resistance)
            / resistance
            * 100
        )

        if distance >= BREAK_CONFIRM_PERCENT:

            candle_confirmed = (
                float(previous["close"])
                > resistance
            )

            momentum_confirmed = (
                momentum > 0
            )

            macd_confirmed = (
                macd > macd_signal
            )

            if (
                candle_confirmed
                and momentum_confirmed
                and macd_confirmed
            ):

                breakout = True

            else:

                false_breakout = True

    # =====================================================
    # BREAKDOWN
    # =====================================================

    if price < support:

        distance = (
            (support - price)
            / support
            * 100
        )

        if distance >= BREAK_CONFIRM_PERCENT:

            candle_confirmed = (
                float(previous["close"])
                < support
            )

            momentum_confirmed = (
                momentum < 0
            )

            macd_confirmed = (
                macd < macd_signal
            )

            if (
                candle_confirmed
                and momentum_confirmed
                and macd_confirmed
            ):

                breakdown = True

            else:

                false_breakdown = True

    return (
        breakout,
        breakdown,
        false_breakout,
        false_breakdown
    )


# =========================================================
# MARKET STATUS
# =========================================================

def market_status(meta):

    state = meta.get(
        "marketState"
    )

    if state == "REGULAR":
        return "LIVE"

    if state == "PRE":
        return "PRE"

    if state == "POST":
        return "POST"

    if state == "CLOSED":
        return "CLOSED"

    return "UNKNOWN"


# =========================================================
# TRADE PLAN
# =========================================================

def build_trade_plan(
    signal,
    price,
    support,
    resistance,
    atr,
    momentum
):

    plan = {
        "trade_status": "NO TRADE",
        "entry": None,
        "stop_loss": None,
        "target_1": None,
        "target_2": None,
        "risk_points": None,
        "reward_1_points": None,
        "reward_2_points": None,
        "risk_reward_1": None,
        "risk_reward_2": None,
        "trailing_stop": None
    }

    if signal not in [
        "BUY",
        "STRONG BUY",
        "SELL",
        "STRONG SELL"
    ]:

        return plan

    if atr <= 0:
        return plan

    # =====================================================
    # BUY
    # =====================================================

    if signal in [
        "BUY",
        "STRONG BUY"
    ]:

        entry = price

        structural_sl = (
            support - atr * 0.20
        )

        volatility_sl = (
            entry - atr * 1.20
        )

        stop_loss = min(
            structural_sl,
            volatility_sl
        )

        risk = entry - stop_loss

        if risk <= 0:
            return plan

        target_1 = entry + (
            risk * 1.50
        )

        target_2 = entry + (
            risk * 2.50
        )

        if resistance > entry:

            target_1 = max(
                target_1,
                resistance
            )

        rr1 = (
            target_1 - entry
        ) / risk

        rr2 = (
            target_2 - entry
        ) / risk

        return {
            "trade_status": "BUY SETUP",
            "entry": safe_round(entry),
            "stop_loss": safe_round(stop_loss),
            "target_1": safe_round(target_1),
            "target_2": safe_round(target_2),
            "risk_points": safe_round(risk),
            "reward_1_points": safe_round(
                target_1 - entry
            ),
            "reward_2_points": safe_round(
                target_2 - entry
            ),
            "risk_reward_1": safe_round(
                rr1,
                2
            ),
            "risk_reward_2": safe_round(
                rr2,
                2
            ),
            "trailing_stop": safe_round(
                entry - atr
            )
        }

    # =====================================================
    # SELL
    # =====================================================

    entry = price

    structural_sl = (
        resistance + atr * 0.20
    )

    volatility_sl = (
        entry + atr * 1.20
    )

    stop_loss = max(
        structural_sl,
        volatility_sl
    )

    risk = stop_loss - entry

    if risk <= 0:
        return plan

    target_1 = entry - (
        risk * 1.50
    )

    target_2 = entry - (
        risk * 2.50
    )

    if support < entry:

        target_1 = min(
            target_1,
            support
        )

    rr1 = (
        entry - target_1
    ) / risk

    rr2 = (
        entry - target_2
    ) / risk

    return {
        "trade_status": "SELL SETUP",
        "entry": safe_round(entry),
        "stop_loss": safe_round(stop_loss),
        "target_1": safe_round(target_1),
        "target_2": safe_round(target_2),
        "risk_points": safe_round(risk),
        "reward_1_points": safe_round(
            entry - target_1
        ),
        "reward_2_points": safe_round(
            entry - target_2
        ),
        "risk_reward_1": safe_round(
            rr1,
            2
        ),
        "risk_reward_2": safe_round(
            rr2,
            2
        ),
        "trailing_stop": safe_round(
            entry + atr
        )
    }


# =========================================================
# HOME
# =========================================================

@app.get("/")
def home():

    return {
        "status": "NIFTY AI backend running",
        "version": ANALYSIS_VERSION,
        "history_endpoint": "/nifty/history"
    }


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "version": ANALYSIS_VERSION
    }


# =========================================================
# LIVE HISTORY / CANDLE ENDPOINT
# =========================================================

@app.get("/nifty/history")
def nifty_history(
    interval: str = Query(
        "5m",
        pattern="^(1m|2m|5m|15m|30m|60m|90m|1h|1d)$"
    )
):

    try:

        # Yahoo limitations:
        # 1m data is generally available only
        # for a short recent period.
        if interval == "1m":

            range_value = "1d"

        elif interval in [
            "2m",
            "5m",
            "15m",
            "30m"
        ]:

            range_value = "5d"

        elif interval in [
            "60m",
            "90m",
            "1h"
        ]:

            range_value = "1mo"

        else:

            range_value = "1y"

        df, meta = fetch_yahoo(
            range_value,
            interval
        )

        if df.empty:

            return {
                "error": "No historical data available",
                "symbol": "NIFTY 50",
                "interval": interval,
                "candles": []
            }

        # Keep response reasonably small
        # for Android mobile.
        max_candles = 300

        df = df.tail(
            max_candles
        ).copy()

        candles = []

        for _, row in df.iterrows():

            candles.append({
                "time": row["time"].isoformat(),
                "open": safe_round(
                    row["open"]
                ),
                "high": safe_round(
                    row["high"]
                ),
                "low": safe_round(
                    row["low"]
                ),
                "close": safe_round(
                    row["close"]
                ),
                "volume": safe_round(
                    row["volume"],
                    0
                )
            })

        return {
            "symbol": "NIFTY 50",
            "interval": interval,
            "candles": candles,
            "count": len(candles),
            "market_status": market_status(meta),
            "latest_price": safe_round(
                float(df["close"].iloc[-1])
            ),
            "time": datetime.now(
                timezone.utc
            ).isoformat(),
            "analysis_version":
                ANALYSIS_VERSION
        }

    except Exception as e:

        return {
            "error": str(e),
            "symbol": "NIFTY 50",
            "interval": interval,
            "candles": [],
            "analysis_version":
                ANALYSIS_VERSION
        }


# =========================================================
# NIFTY ANALYSIS
# =========================================================

@app.get("/nifty")
def nifty_analysis():

    try:

        # =================================================
        # 5 MIN DATA
        # =================================================

        df, meta = fetch_yahoo(
            "1d",
            "5m"
        )

        if len(df) < 30:

            return {
                "error": "Not enough market data",
                "analysis_version":
                    ANALYSIS_VERSION
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

        ma5 = float(
            close.rolling(5)
            .mean()
            .iloc[-1]
        )

        ma10 = float(
            close.rolling(10)
            .mean()
            .iloc[-1]
        )

        ma20 = float(
            close.rolling(20)
            .mean()
            .iloc[-1]
        )

        # =================================================
        # RSI
        # =================================================

        rsi_series = calculate_rsi(
            close,
            14
        )

        rsi14 = float(
            rsi_series.iloc[-1]
        )

        # =================================================
        # MACD
        # =================================================

        (
            macd,
            macd_signal,
            macd_hist
        ) = calculate_macd(close)

        # =================================================
        # MOMENTUM
        # =================================================

        if len(df) >= 6:

            momentum = (
                price -
                float(close.iloc[-6])
            )

        else:

            momentum = 0.0

        # =================================================
        # ATR
        # =================================================

        atr = calculate_atr(
            df,
            ATR_PERIOD
        )

        # =================================================
        # SUPPORT / RESISTANCE
        # =================================================

        support, resistance = (
            calculate_levels(df)
        )

        if (
            support is None
            or
            resistance is None
        ):

            return {
                "error":
                    "Unable to calculate levels",
                "analysis_version":
                    ANALYSIS_VERSION
            }

        support_distance = (
            (price - support)
            / price
            * 100
        )

        resistance_distance = (
            (resistance - price)
            / price
            * 100
        )

        near_support = (
            support_distance
            <= NEAR_LEVEL_PERCENT
        )

        near_resistance = (
            resistance_distance
            <= NEAR_LEVEL_PERCENT
        )

        # =================================================
        # VOLUME
        # =================================================

        (
            volume_status,
            volume_ratio
        ) = calculate_volume(df)

        # =================================================
        # BREAKOUT / BREAKDOWN
        # =================================================

        (
            breakout_confirmed,
            breakdown_confirmed,
            false_breakout,
            false_breakdown
        ) = detect_breakout_breakdown(
            df,
            support,
            resistance,
            momentum,
            macd,
            macd_signal
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
                calculate_trend(
                    df_15m
                )
            )

        except Exception:

            higher_tf_warning = (
                "Higher timeframe data unavailable"
            )

        # =================================================
        # SCORE ENGINE
        # =================================================

        bullish_score = 0
        bearish_score = 0

        reasons = []

        # =================================================
        # SHORT TERM MA
        # =================================================

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

        # =================================================
        # MA20
        # =================================================

        if price > ma20:

            bullish_score += 1

        elif price < ma20:

            bearish_score += 1

        # =================================================
        # HIGHER TIMEFRAME
        # =================================================

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

        if macd > macd_signal:

            bullish_score += 2

            reasons.append(
                "MACD bullish"
            )

        elif macd < macd_signal:

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
        # BREAKOUT
        # =================================================

        if breakout_confirmed:

            bullish_score += 3

            reasons.append(
                "Confirmed resistance breakout"
            )

        # =================================================
        # BREAKDOWN
        # =================================================

        if breakdown_confirmed:

            bearish_score += 3

            reasons.append(
                "Confirmed support breakdown"
            )

        # =================================================
        # HTF CONFLICT
        # =================================================

        higher_tf_conflict = False

        if (
            higher_tf_trend == "BULLISH"
            and
            bearish_score >
            bullish_score
        ):

            higher_tf_conflict = True

        elif (
            higher_tf_trend == "BEARISH"
            and
            bullish_score >
            bearish_score
        ):

            higher_tf_conflict = True

        # =================================================
        # SIGNAL
        # =================================================

        signal = "NEUTRAL"
        signal_strength = "LOW"

        if breakdown_confirmed:

            signal = "STRONG SELL"
            signal_strength = "HIGH"

        elif breakout_confirmed:

            signal = "STRONG BUY"
            signal_strength = "HIGH"

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

        elif higher_tf_conflict:

            signal = "WAIT"
            signal_strength = "HIGH"

            reasons.append(
                "Higher timeframe conflict"
            )

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

        elif (
            bullish_score >= MIN_SCORE
            and
            bullish_score >
            bearish_score +
            MIN_SCORE_GAP
        ):

            signal = "BUY"
            signal_strength = "MODERATE"

        elif (
            bearish_score >= MIN_SCORE
            and
            bearish_score >
            bullish_score +
            MIN_SCORE_GAP
        ):

            signal = "SELL"
            signal_strength = "MODERATE"

        # =================================================
        # CONFIDENCE
        # =================================================

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

            confidence = 50.0

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

        else:

            confidence = min(
                confidence,
                92
            )

        # =================================================
        # TRADE PLAN
        # =================================================

        trade_plan = build_trade_plan(
            signal,
            price,
            support,
            resistance,
            atr,
            momentum
        )

        # =================================================
        # WARNINGS
        # =================================================

        warnings = []

        if volume_status == "UNAVAILABLE":

            warnings.append(
                "Volume data unavailable"
            )

        if higher_tf_warning:

            warnings.append(
                higher_tf_warning
            )

        if market_status(meta) != "LIVE":

            warnings.append(
                "Market is not currently LIVE"
            )

        if signal in [
            "BUY",
            "STRONG BUY",
            "SELL",
            "STRONG SELL"
        ]:

            rr_value = trade_plan[
                "risk_reward_1"
            ]

            if (
                rr_value is not None
                and
                rr_value < MIN_RR
            ):

                warnings.append(
                    "Risk/Reward below preferred threshold"
                )

        # =================================================
        # FINAL TREND
        # =================================================

        if bullish_score > bearish_score:

            final_trend = "BULLISH"

        elif bearish_score > bullish_score:

            final_trend = "BEARISH"

        else:

            final_trend = "SIDEWAYS"

        # =================================================
        # ENTRY / AVOID ZONE
        # =================================================

        entry_zone = None
        avoid_zone = None

        if signal in [
            "BUY",
            "STRONG BUY"
        ]:

            entry_zone = [
                safe_round(price),
                safe_round(
                    max(
                        price,
                        resistance
                    )
                )
            ]

            avoid_zone = [
                safe_round(support),
                safe_round(
                    support - atr
                )
            ]

        elif signal in [
            "SELL",
            "STRONG SELL"
        ]:

            entry_zone = [
                safe_round(
                    min(
                        price,
                        support
                    )
                ),
                safe_round(price)
            ]

            avoid_zone = [
                safe_round(resistance),
                safe_round(
                    resistance + atr
                )
            ]

        elif signal == "WAIT":

            avoid_zone = [
                safe_round(support),
                safe_round(resistance)
            ]

        # =================================================
        # RESPONSE
        # =================================================

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
                market_status(meta),

            "trend":
                final_trend,

            "higher_timeframe_trend":
                higher_tf_trend,

            "higher_tf_conflict":
                higher_tf_conflict,

            "signal":
                signal,

            "signal_strength":
                signal_strength,

            "confidence":
                safe_round(confidence),

            "bullish_score":
                bullish_score,

            "bearish_score":
                bearish_score,

            "rsi_14":
                safe_round(rsi14),

            "ma_5":
                safe_round(ma5),

            "ma_10":
                safe_round(ma10),

            "ma_20":
                safe_round(ma20),

            "macd":
                safe_round(macd),

            "macd_signal":
                safe_round(macd_signal),

            "macd_histogram":
                safe_round(macd_hist),

            "momentum":
                safe_round(momentum),

            "atr_14":
                safe_round(atr),

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

            "breakout_confirmed":
                breakout_confirmed,

            "breakdown_confirmed":
                breakdown_confirmed,

            "false_breakout":
                false_breakout,

            "false_breakdown":
                false_breakdown,

            "volume_status":
                volume_status,

            "volume_ratio":
                safe_round(
                    volume_ratio,
                    2
                ),

            "entry_zone":
                entry_zone,

            "avoid_zone":
                avoid_zone,

            "trade_status":
                trade_plan[
                    "trade_status"
                ],

            "entry":
                trade_plan["entry"],

            "stop_loss":
                trade_plan["stop_loss"],

            "target_1":
                trade_plan["target_1"],

            "target_2":
                trade_plan["target_2"],

            "risk_points":
                trade_plan["risk_points"],

            "reward_1_points":
                trade_plan[
                    "reward_1_points"
                ],

            "reward_2_points":
                trade_plan[
                    "reward_2_points"
                ],

            "risk_reward_1":
                trade_plan[
                    "risk_reward_1"
                ],

            "risk_reward_2":
                trade_plan[
                    "risk_reward_2"
                ],

            "trailing_stop":
                trade_plan[
                    "trailing_stop"
                ],

            "reasons":
                reasons,

            "warnings":
                warnings,

            "analysis_version":
                ANALYSIS_VERSION,

            "time":
                datetime.now(
                    timezone.utc
                ).isoformat()
        }

    except Exception as e:

        return {

            "error": str(e),

            "analysis_version":
                ANALYSIS_VERSION
        }
