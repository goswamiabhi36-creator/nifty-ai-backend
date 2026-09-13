from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import math
import time
from datetime import datetime, timedelta, timezone, time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Nifty AI Trader Backend",
    version="4.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# CONFIG
# ============================================================

SYMBOL = "^NSEI"

IST = ZoneInfo("Asia/Kolkata")

MARKET_OPEN = dt_time(9, 15)
MARKET_CLOSE = dt_time(15, 30)

# We do NOT pretend old data is live.
# Maximum acceptable age for LIVE status.
MAX_LIVE_AGE_SECONDS = 90

# Yahoo intraday limits can vary.
INTERVAL_PERIODS = {
    "1m": "1d",
    "5m": "5d",
    "15m": "1mo",
    "30m": "1mo",
    "1h": "3mo",
    "1D": "1y",
}


# ============================================================
# BASIC HELPERS
# ============================================================

def now_ist():
    return datetime.now(IST)


def is_weekday(dt):
    return dt.weekday() < 5


def is_market_hours(dt):
    if not is_weekday(dt):
        return False

    current = dt.time()

    return MARKET_OPEN <= current <= MARKET_CLOSE


def safe_float(value):
    try:
        if value is None:
            return None

        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return None

        return value

    except Exception:
        return None


def round_or_none(value, digits=2):
    value = safe_float(value)

    if value is None:
        return None

    return round(value, digits)


def normalize_columns(df):
    """
    yfinance may return normal columns or MultiIndex columns.
    Convert both formats into simple OHLCV columns.
    """

    if df is None or df.empty:
        return df

    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):
        new_columns = []

        for col in result.columns:
            if isinstance(col, tuple):
                new_columns.append(str(col[0]))
            else:
                new_columns.append(str(col))

        result.columns = new_columns

    result.columns = [
        str(c).strip().lower()
        for c in result.columns
    ]

    rename_map = {
        "adj close": "adj_close"
    }

    result = result.rename(columns=rename_map)

    required = ["open", "high", "low", "close"]

    for column in required:
        if column not in result.columns:
            raise ValueError(
                f"Missing required column: {column}"
            )

    if "volume" not in result.columns:
        result["volume"] = np.nan

    return result


# ============================================================
# DATA FETCH
# ============================================================

def fetch_history(interval="5m"):
    if interval not in INTERVAL_PERIODS:
        interval = "5m"

    period = INTERVAL_PERIODS[interval]

    try:
        df = yf.download(
            SYMBOL,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False
        )

    except Exception as exc:
        raise RuntimeError(
            f"Market data download failed: {exc}"
        )

    if df is None or df.empty:
        raise RuntimeError(
            "No market data returned by data provider."
        )

    df = normalize_columns(df)

    # Remove rows without close
    df = df.dropna(subset=["close"])

    if df.empty:
        raise RuntimeError(
            "Market data contains no valid candles."
        )

    return df


def get_latest_timestamp(df):
    if df is None or df.empty:
        return None

    ts = df.index[-1]

    try:
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")

        return ts.to_pydatetime().astimezone(IST)

    except Exception:
        return None


def get_data_age_seconds(df):
    latest = get_latest_timestamp(df)

    if latest is None:
        return None

    current = now_ist()

    age = (current - latest).total_seconds()

    # Prevent negative age caused by clock differences.
    return max(0.0, age)


# ============================================================
# MARKET STATUS
# ============================================================

def calculate_market_status(df):
    """
    IMPORTANT:
    LIVE means:
      1. Current time is during Indian market hours
      2. Latest candle is recent enough

    We do not simply assume LIVE.
    """

    current = now_ist()

    market_session = is_market_hours(current)

    latest_timestamp = get_latest_timestamp(df)

    age_seconds = get_data_age_seconds(df)

    if latest_timestamp is None:
        return {
            "market_status": "UNKNOWN",
            "data_age_seconds": None,
            "latest_data_time": None
        }

    if not market_session:
        return {
            "market_status": "CLOSED",
            "data_age_seconds": round(age_seconds, 1)
            if age_seconds is not None else None,
            "latest_data_time": latest_timestamp.isoformat()
        }

    if age_seconds is None:
        return {
            "market_status": "UNKNOWN",
            "data_age_seconds": None,
            "latest_data_time": latest_timestamp.isoformat()
        }

    if age_seconds <= MAX_LIVE_AGE_SECONDS:
        status = "LIVE"
    else:
        status = "DELAYED"

    return {
        "market_status": status,
        "data_age_seconds": round(age_seconds, 1),
        "latest_data_time": latest_timestamp.isoformat()
    }


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df):
    result = df.copy()

    close = result["close"]
    high = result["high"]
    low = result["low"]

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    result["ma_5"] = close.rolling(5).mean()
    result["ma_10"] = close.rolling(10).mean()
    result["ma_20"] = close.rolling(20).mean()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    result["ema_9"] = close.ewm(
        span=9,
        adjust=False
    ).mean()

    result["ema_20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    result["ema_50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    result["ema_100"] = close.ewm(
        span=100,
        adjust=False
    ).mean()

    result["ema_200"] = close.ewm(
        span=200,
        adjust=False
    ).mean()

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result["rsi_14"] = 100 - (
        100 / (1 + rs)
    )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    ema12 = close.ewm(
        span=12,
        adjust=False
    ).mean()

    ema26 = close.ewm(
        span=26,
        adjust=False
    ).mean()

    result["macd"] = ema12 - ema26

    result["macd_signal"] = result["macd"].ewm(
        span=9,
        adjust=False
    ).mean()

    result["macd_histogram"] = (
        result["macd"] -
        result["macd_signal"]
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    result["momentum"] = close.diff(10)

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat(
        [tr1, tr2, tr3],
        axis=1
    ).max(axis=1)

    result["atr_14"] = true_range.rolling(14).mean()

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where(
        (up_move > down_move) & (up_move > 0),
        up_move,
        0
    )

    minus_dm = np.where(
        (down_move > up_move) & (down_move > 0),
        down_move,
        0
    )

    atr14 = result["atr_14"]

    plus_di = (
        100 *
        pd.Series(
            plus_dm,
            index=result.index
        ).rolling(14).mean() /
        atr14
    )

    minus_di = (
        100 *
        pd.Series(
            minus_dm,
            index=result.index
        ).rolling(14).mean() /
        atr14
    )

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(0, np.nan)
    )

    result["adx_14"] = dx.rolling(14).mean()

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    typical_price = (
        high +
        low +
        close
    ) / 3

    volume = result["volume"].fillna(0)

    cumulative_volume = volume.cumsum()

    cumulative_value = (
        typical_price * volume
    ).cumsum()

    result["vwap"] = np.where(
        cumulative_volume > 0,
        cumulative_value / cumulative_volume,
        typical_price
    )

    result["vwap"] = pd.Series(
        result["vwap"],
        index=result.index
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    middle = close.rolling(20).mean()
    std = close.rolling(20).std()

    result["bollinger_middle"] = middle

    result["bollinger_upper"] = (
        middle + 2 * std
    )

    result["bollinger_lower"] = (
        middle - 2 * std
    )

    return result


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def calculate_support_resistance(df):
    recent = df.tail(50)

    support = safe_float(
        recent["low"].rolling(10).min().iloc[-1]
    )

    resistance = safe_float(
        recent["high"].rolling(10).max().iloc[-1]
    )

    support2 = safe_float(
        recent["low"].rolling(20).min().iloc[-1]
    )

    resistance2 = safe_float(
        recent["high"].rolling(20).max().iloc[-1]
    )

    return (
        support,
        support2,
        resistance,
        resistance2
    )


# ============================================================
# VOLUME
# ============================================================

def calculate_volume_status(df):
    volume = df["volume"].dropna()

    if volume.empty:
        return "UNAVAILABLE", None

    if len(volume) < 20:
        return "UNAVAILABLE", None

    latest_volume = safe_float(volume.iloc[-1])

    average_volume = safe_float(
        volume.tail(20).mean()
    )

    if (
        latest_volume is None or
        average_volume is None or
        average_volume <= 0
    ):
        return "UNAVAILABLE", None

    ratio = latest_volume / average_volume

    if ratio >= 1.5:
        status = "HIGH"

    elif ratio <= 0.7:
        status = "LOW"

    else:
        status = "NORMAL"

    return status, round(ratio, 2)


# ============================================================
# TREND
# ============================================================

def calculate_trend(row):
    close = safe_float(row["close"])

    ema9 = safe_float(row["ema_9"])
    ema20 = safe_float(row["ema_20"])
    ema50 = safe_float(row["ema_50"])

    if None in (
        close,
        ema9,
        ema20,
        ema50
    ):
        return "SIDEWAYS"

    bullish = (
        close > ema20 and
        ema9 > ema20 and
        ema20 > ema50
    )

    bearish = (
        close < ema20 and
        ema9 < ema20 and
        ema20 < ema50
    )

    if bullish:
        return "BULLISH"

    if bearish:
        return "BEARISH"

    return "SIDEWAYS"


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(
    row,
    market_status,
    support,
    resistance,
    support2,
    resistance2
):

    close = safe_float(row["close"])

    ema9 = safe_float(row["ema_9"])
    ema20 = safe_float(row["ema_20"])
    ema50 = safe_float(row["ema_50"])

    rsi = safe_float(row["rsi_14"])

    macd = safe_float(row["macd"])
    macd_signal = safe_float(row["macd_signal"])

    momentum = safe_float(row["momentum"])

    vwap = safe_float(row["vwap"])

    adx = safe_float(row["adx_14"])

    if close is None:
        return {
            "signal": "NEUTRAL",
            "signal_strength": "LOW",
            "confidence": 0,
            "bullish_score": 0,
            "bearish_score": 0,
            "reasons": [],
            "warnings": [
                "No valid price data"
            ]
        }

    bullish_score = 0
    bearish_score = 0

    bullish_reasons = []
    bearish_reasons = []

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    if ema9 is not None and ema20 is not None:

        if ema9 > ema20:
            bullish_score += 1
            bullish_reasons.append(
                "EMA9 above EMA20"
            )

        elif ema9 < ema20:
            bearish_score += 1
            bearish_reasons.append(
                "EMA9 below EMA20"
            )

    if ema20 is not None and ema50 is not None:

        if ema20 > ema50:
            bullish_score += 1
            bullish_reasons.append(
                "EMA20 above EMA50"
            )

        elif ema20 < ema50:
            bearish_score += 1
            bearish_reasons.append(
                "EMA20 below EMA50"
            )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi is not None:

        if rsi >= 55:
            bullish_score += 2
            bullish_reasons.append(
                "RSI bullish"
            )

        elif rsi <= 45:
            bearish_score += 2
            bearish_reasons.append(
                "RSI bearish"
            )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if (
        macd is not None and
        macd_signal is not None
    ):

        if macd > macd_signal:
            bullish_score += 2
            bullish_reasons.append(
                "MACD bullish"
            )

        else:
            bearish_score += 2
            bearish_reasons.append(
                "MACD bearish"
            )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    if vwap is not None:

        if close > vwap:
            bullish_score += 2
            bullish_reasons.append(
                "Price above VWAP"
            )

        else:
            bearish_score += 2
            bearish_reasons.append(
                "Price below VWAP"
            )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    if momentum is not None:

        if momentum > 0:
            bullish_score += 1
            bullish_reasons.append(
                "Positive momentum"
            )

        elif momentum < 0:
            bearish_score += 1
            bearish_reasons.append(
                "Negative momentum"
            )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    strong_trend = False

    if adx is not None:
        strong_trend = adx >= 25

    # --------------------------------------------------------
    # Breakout / Breakdown
    # --------------------------------------------------------

    breakout_confirmed = False
    breakdown_confirmed = False

    if resistance is not None:

        if close > resistance:
            breakout_confirmed = True
            bullish_score += 3
            bullish_reasons.append(
                "Resistance breakout"
            )

    if support is not None:

        if close < support:
            breakdown_confirmed = True
            bearish_score += 3
            bearish_reasons.append(
                "Support breakdown"
            )

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    difference = abs(
        bullish_score -
        bearish_score
    )

    total_score = (
        bullish_score +
        bearish_score
    )

    if total_score <= 0:
        confidence = 50.0
    else:
        confidence = (
            max(
                bullish_score,
                bearish_score
            ) /
            total_score
        ) * 100

    confidence = round(
        max(0, min(100, confidence)),
        1
    )

    signal = "NEUTRAL"
    strength = "LOW"

    # --------------------------------------------------------
    # IMPORTANT:
    # Do not issue BUY/SELL when data isn't LIVE.
    # --------------------------------------------------------

    if market_status != "LIVE":

        signal = "NEUTRAL"
        strength = "LOW"

    else:

        if (
            bullish_score >= 7 and
            difference >= 3
        ):
            signal = "BUY"

            strength = (
                "HIGH"
                if difference >= 5
                else "MEDIUM"
            )

        elif (
            bearish_score >= 7 and
            difference >= 3
        ):
            signal = "SELL"

            strength = (
                "HIGH"
                if difference >= 5
                else "MEDIUM"
            )

    # --------------------------------------------------------
    # Correct trend calculation
    # --------------------------------------------------------

    trend = calculate_trend(row)

    # DO NOT add incorrect "Strong bullish trend"
    # when trend is SIDEWAYS.

    if trend == "BULLISH":
        trend_reason = "Bullish trend"

    elif trend == "BEARISH":
        trend_reason = "Bearish trend"

    else:
        trend_reason = "Sideways trend"

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    reasons = []

    reasons.extend(
        bullish_reasons
        if bullish_score >= bearish_score
        else bearish_reasons
    )

    if not reasons:
        reasons = [trend_reason]

    warnings = []

    if market_status != "LIVE":
        warnings.append(
            f"Market data status: {market_status}"
        )

    return {
        "signal": signal,
        "signal_strength": strength,
        "confidence": confidence,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "reasons": reasons,
        "warnings": warnings,
        "breakout_confirmed": breakout_confirmed,
        "breakdown_confirmed": breakdown_confirmed,
        "trend": trend
    }


# ============================================================
# TRADE PLAN
# ============================================================

def create_trade_plan(
    signal_data,
    row,
    support,
    resistance
):

    signal = signal_data["signal"]

    close = safe_float(row["close"])
    atr = safe_float(row["atr_14"])

    # Never create trade plan without valid values.
    if (
        signal not in ("BUY", "SELL") or
        close is None or
        atr is None or
        atr <= 0
    ):

        return {
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
            "trailing_stop": None,
            "entry_zone": None,
            "avoid_zone": None
        }

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if signal == "BUY":

        entry = close

        stop_loss = entry - (
            atr * 1.2
        )

        target1 = entry + (
            atr * 1.8
        )

        target2 = entry + (
            atr * 2.8
        )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    else:

        entry = close

        stop_loss = entry + (
            atr * 1.2
        )

        target1 = entry - (
            atr * 1.8
        )

        target2 = entry - (
            atr * 2.8
        )

    risk = abs(
        entry -
        stop_loss
    )

    reward1 = abs(
        target1 -
        entry
    )

    reward2 = abs(
        target2 -
        entry
    )

    rr1 = (
        reward1 / risk
        if risk > 0
        else None
    )

    rr2 = (
        reward2 / risk
        if risk > 0
        else None
    )

    entry_zone = (
        f"{entry - atr * 0.15:.2f} - "
        f"{entry + atr * 0.15:.2f}"
    )

    avoid_zone = None

    return {
        "trade_status": "TRADE SETUP",
        "entry": round(entry, 2),
        "stop_loss": round(stop_loss, 2),
        "target_1": round(target1, 2),
        "target_2": round(target2, 2),
        "risk_points": round(risk, 2),
        "reward_1_points": round(reward1, 2),
        "reward_2_points": round(reward2, 2),
        "risk_reward_1": round(rr1, 2)
        if rr1 is not None else None,
        "risk_reward_2": round(rr2, 2)
        if rr2 is not None else None,
        "trailing_stop": round(
            atr * 1.0,
            2
        ),
        "entry_zone": entry_zone,
        "avoid_zone": avoid_zone
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def build_analysis(interval="5m"):

    df = fetch_history(interval)

    df = calculate_indicators(df)

    row = df.iloc[-1]

    market_info = calculate_market_status(df)

    market_status = market_info[
        "market_status"
    ]

    support, support2, resistance, resistance2 = (
        calculate_support_resistance(df)
    )

    volume_status, volume_ratio = (
        calculate_volume_status(df)
    )

    signal_data = generate_signal(
        row=row,
        market_status=market_status,
        support=support,
        resistance=resistance,
        support2=support2,
        resistance2=resistance2
    )

    trade_plan = create_trade_plan(
        signal_data=signal_data,
        row=row,
        support=support,
        resistance=resistance
    )

    price = safe_float(row["close"])

    # --------------------------------------------------------
    # Previous close
    # --------------------------------------------------------

    previous_close = None

    if len(df) >= 2:
        previous_close = safe_float(
            df["close"].iloc[-2]
        )

    change = None
    change_percent = None

    if (
        price is not None and
        previous_close is not None and
        previous_close != 0
    ):

        change = price - previous_close

        change_percent = (
            change /
            previous_close
        ) * 100

    # --------------------------------------------------------
    # Higher timeframe
    # --------------------------------------------------------

    higher_tf_trend = "SIDEWAYS"

    try:

        higher_df = fetch_history("15m")

        higher_df = calculate_indicators(
            higher_df
        )

        higher_row = higher_df.iloc[-1]

        higher_tf_trend = calculate_trend(
            higher_row
        )

    except Exception:
        higher_tf_trend = "SIDEWAYS"

    trend = signal_data["trend"]

    higher_tf_conflict = (
        trend != "SIDEWAYS" and
        higher_tf_trend != "SIDEWAYS" and
        trend != higher_tf_trend
    )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    support_distance_percent = None
    resistance_distance_percent = None

    if (
        price is not None and
        support is not None and
        price != 0
    ):
        support_distance_percent = (
            abs(price - support) /
            price
        ) * 100

    if (
        price is not None and
        resistance is not None and
        price != 0
    ):
        resistance_distance_percent = (
            abs(resistance - price) /
            price
        ) * 100

    near_support = (
        support_distance_percent is not None and
        support_distance_percent <= 0.15
    )

    near_resistance = (
        resistance_distance_percent is not None and
        resistance_distance_percent <= 0.15
    )

    # --------------------------------------------------------
    # False breakout detection
    # --------------------------------------------------------

    false_breakout = False
    false_breakdown = False

    if len(df) >= 3:

        previous_candle = df.iloc[-2]
        current_candle = df.iloc[-1]

        prev_close = safe_float(
            previous_candle["close"]
        )

        current_close = safe_float(
            current_candle["close"]
        )

        if (
            resistance is not None and
            prev_close is not None and
            current_close is not None
        ):

            if (
                prev_close > resistance and
                current_close < resistance
            ):
                false_breakout = True

        if (
            support is not None and
            prev_close is not None and
            current_close is not None
        ):

            if (
                prev_close < support and
                current_close > support
            ):
                false_breakdown = True

    # --------------------------------------------------------
    # Warnings
    # --------------------------------------------------------

    warnings = list(
        signal_data.get(
            "warnings",
            []
        )
    )

    if volume_status == "UNAVAILABLE":

        if "Volume data unavailable" not in warnings:
            warnings.append(
                "Volume data unavailable"
            )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    response = {

        "symbol": "NIFTY 50",

        "price": round_or_none(price),

        "previous_close": round_or_none(
            previous_close
        ),

        "change": round_or_none(change),

        "change_percent": round_or_none(
            change_percent
        ),

        "market_status": market_status,

        "data_age_seconds":
            market_info["data_age_seconds"],

        "latest_data_time":
            market_info["latest_data_time"],

        "trend": trend,

        "higher_timeframe_trend":
            higher_tf_trend,

        "higher_tf_conflict":
            higher_tf_conflict,

        "signal":
            signal_data["signal"],

        "signal_strength":
            signal_data["signal_strength"],

        "confidence":
            signal_data["confidence"],

        "bullish_score":
            signal_data["bullish_score"],

        "bearish_score":
            signal_data["bearish_score"],

        # ----------------------------------------------------
        # Moving averages
        # ----------------------------------------------------

        "ma_5": round_or_none(
            row["ma_5"]
        ),

        "ma_10": round_or_none(
            row["ma_10"]
        ),

        "ma_20": round_or_none(
            row["ma_20"]
        ),

        "ema_9": round_or_none(
            row["ema_9"]
        ),

        "ema_20": round_or_none(
            row["ema_20"]
        ),

        "ema_50": round_or_none(
            row["ema_50"]
        ),

        "ema_100": round_or_none(
            row["ema_100"]
        ),

        "ema_200": round_or_none(
            row["ema_200"]
        ),

        # ----------------------------------------------------
        # Indicators
        # ----------------------------------------------------

        "rsi_14": round_or_none(
            row["rsi_14"]
        ),

        "macd": round_or_none(
            row["macd"]
        ),

        "macd_signal": round_or_none(
            row["macd_signal"]
        ),

        "macd_histogram": round_or_none(
            row["macd_histogram"]
        ),

        "momentum": round_or_none(
            row["momentum"]
        ),

        "atr_14": round_or_none(
            row["atr_14"]
        ),

        "adx_14": round_or_none(
            row["adx_14"]
        ),

        "vwap": round_or_none(
            row["vwap"]
        ),

        # ----------------------------------------------------
        # Bollinger
        # ----------------------------------------------------

        "bollinger_upper":
            round_or_none(
                row["bollinger_upper"]
            ),

        "bollinger_middle":
            round_or_none(
                row["bollinger_middle"]
            ),

        "bollinger_lower":
            round_or_none(
                row["bollinger_lower"]
            ),

        # ----------------------------------------------------
        # Support resistance
        # ----------------------------------------------------

        "support":
            round_or_none(support),

        "support_2":
            round_or_none(support2),

        "resistance":
            round_or_none(resistance),

        "resistance_2":
            round_or_none(resistance2),

        "support_distance_percent":
            round_or_none(
                support_distance_percent,
                3
            ),

        "resistance_distance_percent":
            round_or_none(
                resistance_distance_percent,
                3
            ),

        "near_support":
            near_support,

        "near_resistance":
            near_resistance,

        # ----------------------------------------------------
        # Breakout
        # ----------------------------------------------------

        "breakout_confirmed":
            signal_data["breakout_confirmed"],

        "breakdown_confirmed":
            signal_data["breakdown_confirmed"],

        "false_breakout":
            false_breakout,

        "false_breakdown":
            false_breakdown,

        # ----------------------------------------------------
        # Volume
        # ----------------------------------------------------

        "volume_status":
            volume_status,

        "volume_ratio":
            volume_ratio,

        # ----------------------------------------------------
        # Trade plan
        # ----------------------------------------------------

        "entry_zone":
            trade_plan["entry_zone"],

        "avoid_zone":
            trade_plan["avoid_zone"],

        "trade_status":
            trade_plan["trade_status"],

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
            trade_plan["reward_1_points"],

        "reward_2_points":
            trade_plan["reward_2_points"],

        "risk_reward_1":
            trade_plan["risk_reward_1"],

        "risk_reward_2":
            trade_plan["risk_reward_2"],

        "trailing_stop":
            trade_plan["trailing_stop"],

        # ----------------------------------------------------
        # AI reasons
        # ----------------------------------------------------

        "reasons":
            signal_data["reasons"],

        "warnings":
            warnings,

        "analysis_version":
            "V4.0",

        "time":
            datetime.now(
                timezone.utc
            ).isoformat()
    }

    return response


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def root():

    return {
        "app": "Nifty AI Trader Backend",
        "version": "4.0",
        "status": "running",
        "symbol": "NIFTY 50",
        "data_source": "Yahoo Finance",
        "live_detection": True,
        "message":
            "Backend is running. "
            "Use /nifty for analysis."
    }


@app.get("/health")
def health():

    return {
        "status": "ok",
        "time":
            datetime.now(
                timezone.utc
            ).isoformat()
    }


@app.get("/nifty")
def nifty():

    try:

        return build_analysis("5m")

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )


@app.get("/nifty/{interval}")
def nifty_interval(interval: str):

    allowed = [
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "1D"
    ]

    if interval not in allowed:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid interval. "
                "Use 1m, 5m, 15m, "
                "30m, 1h or 1D."
            )
        )

    try:

        return build_analysis(interval)

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )


# ============================================================
# HISTORY ENDPOINT
# ============================================================

@app.get("/nifty/history")
def nifty_history(
    interval: str = "5m"
):

    allowed = [
        "1m",
        "5m",
        "15m",
        "30m",
        "1h",
        "1D"
    ]

    if interval not in allowed:
        raise HTTPException(
            status_code=400,
            detail="Invalid interval."
        )

    try:

        df = fetch_history(interval)

        candles = []

        for index, row in df.tail(300).iterrows():

            try:

                timestamp = index

                if timestamp.tzinfo is None:
                    timestamp = timestamp.tz_localize(
                        "UTC"
                    )

                timestamp = timestamp.tz_convert(
                    IST
                )

                timestamp_string = (
                    timestamp.isoformat()
                )

            except Exception:

                timestamp_string = str(index)

            candles.append({

                "timestamp":
                    timestamp_string,

                "open":
                    round_or_none(
                        row["open"]
                    ),

                "high":
                    round_or_none(
                        row["high"]
                    ),

                "low":
                    round_or_none(
                        row["low"]
                    ),

                "close":
                    round_or_none(
                        row["close"]
                    ),

                "volume":
                    round_or_none(
                        row["volume"]
                    )
            })

        age = get_data_age_seconds(df)

        status = calculate_market_status(
            df
        )

        return {

            "symbol": "NIFTY 50",

            "interval": interval,

            "market_status":
                status["market_status"],

            "data_age_seconds":
                status["data_age_seconds"],

            "latest_data_time":
                status["latest_data_time"],

            "count":
                len(candles),

            "candles":
                candles,

            "time":
                datetime.now(
                    timezone.utc
                ).isoformat()
        }

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )


# ============================================================
# RUN LOCALLY
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=10000,
        reload=False
    )
