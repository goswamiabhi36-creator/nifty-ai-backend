from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime, time as dt_time, timezone
from zoneinfo import ZoneInfo
from typing import Optional
import threading
import time
import requests
import os

from kotak_oi import (
    get_oi_analysis,
    kotak_status,
    get_nearest_expiry
)


# ============================================================
# NIFTY AI TRADER — FINAL BACKEND
# Version: FINAL-1.0
# Data Source: Yahoo Finance (^NSEI)
# ============================================================


app = FastAPI(
    title="NIFTY AI TRADER",
    version="FINAL-1.0"
)


# ============================================================
# CORS
# ============================================================

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

MAX_LIVE_AGE_SECONDS = 120


INTERVAL_CONFIG = {
    "1m": "1d",
    "5m": "5d",
    "15m": "1mo",
    "30m": "1mo",
    "1h": "3mo",
    "1D": "1y",
}


# Cache lifetime in seconds
CACHE_TTL = {
    "1m": 10,
    "5m": 10,
    "15m": 30,
    "30m": 60,
    "1h": 120,
    "1D": 300,
}


# ============================================================
# MEMORY CACHE
# ============================================================

_cache = {}
_cache_lock = threading.Lock()


def cache_get(interval: str):
    with _cache_lock:
        item = _cache.get(interval)

        if item is None:
            return None

        timestamp, dataframe = item

        ttl = CACHE_TTL.get(interval, 30)

        if time.time() - timestamp > ttl:
            return None

        return dataframe.copy()


def cache_set(interval: str, dataframe: pd.DataFrame):
    with _cache_lock:
        _cache[interval] = (
            time.time(),
            dataframe.copy()
        )


# ============================================================
# BASIC HELPERS
# ============================================================

def now_ist() -> datetime:
    return datetime.now(IST)


def safe_float(value, digits: int = 2):
    try:
        if value is None:
            return None

        value = float(value)

        if not np.isfinite(value):
            return None

        return round(value, digits)

    except Exception:
        return None


def safe_int(value):
    try:
        if value is None:
            return None

        value = int(value)

        return value

    except Exception:
        return None


def clean_value(value):
    """
    Converts numpy/pandas values to JSON-safe Python values.
    """
    if value is None:
        return None

    if isinstance(value, (np.integer,)):
        return int(value)

    if isinstance(value, (np.floating,)):
        if not np.isfinite(float(value)):
            return None
        return float(value)

    if isinstance(value, float):
        if not np.isfinite(value):
            return None
        return value

    return value


def normalize_number(value):
    value = clean_value(value)

    if value is None:
        return None

    return round(float(value), 2)


# ============================================================
# DATAFRAME NORMALIZATION
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Handle Yahoo Finance MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):

        new_columns = []

        for column in df.columns:

            if isinstance(column, tuple):

                found = None

                for part in column:
                    text = str(part).strip()

                    if text.lower() in {
                        "open",
                        "high",
                        "low",
                        "close",
                        "adj close",
                        "volume"
                    }:
                        found = text
                        break

                if found:
                    new_columns.append(found)
                else:
                    new_columns.append(str(column[0]))

            else:
                new_columns.append(str(column))

        df.columns = new_columns

    rename_map = {}

    for column in df.columns:

        lower = str(column).strip().lower()

        if lower == "open":
            rename_map[column] = "Open"

        elif lower == "high":
            rename_map[column] = "High"

        elif lower == "low":
            rename_map[column] = "Low"

        elif lower == "close":
            rename_map[column] = "Close"

        elif lower == "adj close":
            rename_map[column] = "Adj Close"

        elif lower == "volume":
            rename_map[column] = "Volume"

    df = df.rename(columns=rename_map)

    required = ["Open", "High", "Low", "Close"]

    for column in required:

        if column not in df.columns:
            return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    for column in ["Open", "High", "Low", "Close", "Volume"]:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    df = df.dropna(
        subset=["Open", "High", "Low", "Close"]
    )

    if df.empty:
        return pd.DataFrame()

    # ========================================================
    # TIMEZONE NORMALIZATION
    # ========================================================

    try:

        if df.index.tz is None:

            df.index = df.index.tz_localize(
                "UTC"
            )

        df.index = df.index.tz_convert(IST)

    except Exception:

        try:

            df.index = pd.to_datetime(
                df.index,
                utc=True
            ).tz_convert(IST)

        except Exception:
            pass

    df = df.sort_index()

    return df


# ============================================================
# FETCH HISTORY
# ============================================================

def fetch_history(interval: str) -> pd.DataFrame:

    if interval not in INTERVAL_CONFIG:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported interval: {interval}"
        )

    cached = cache_get(interval)

    if cached is not None:
        return cached

    period = INTERVAL_CONFIG[interval]

    try:

        df = yf.download(
            SYMBOL,
            period=period,
            interval=interval,
            progress=False,
            auto_adjust=False,
            threads=False
        )

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=f"Yahoo Finance data error: {str(exc)}"
        )

    df = normalize_columns(df)

    if df.empty:

        raise HTTPException(
            status_code=503,
            detail=f"No market data available for interval {interval}"
        )

    cache_set(interval, df)

    return df.copy()


# ============================================================
# MARKET STATUS
# ============================================================

def get_market_status(
    latest_timestamp=None
):

    current = now_ist()

    current_time = current.time()

    weekday = current.weekday()

    if weekday >= 5:

        return {
            "status": "CLOSED",
            "message": "Market closed — weekend."
        }

    if current_time < MARKET_OPEN:

        return {
            "status": "PRE_MARKET",
            "message": "Market has not opened yet."
        }

    if current_time > MARKET_CLOSE:

        return {
            "status": "CLOSED",
            "message": "Market closed for today."
        }

    if latest_timestamp is None:

        return {
            "status": "UNKNOWN",
            "message": "Latest market timestamp unavailable."
        }

    try:

        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.replace(
                tzinfo=IST
            )

        age = (
            current - latest_timestamp
        ).total_seconds()

        if age <= MAX_LIVE_AGE_SECONDS:

            return {
                "status": "LIVE",
                "message": "Latest market data available."
            }

        return {
            "status": "DELAYED",
            "message": "Latest market data is delayed."
        }

    except Exception:

        return {
            "status": "UNKNOWN",
            "message": "Unable to determine market status."
        }


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(
    df: pd.DataFrame
):

    data = df.copy()

    close = data["Close"]
    high = data["High"]
    low = data["Low"]

    # --------------------------------------------------------
    # Moving Averages
    # --------------------------------------------------------

    data["MA5"] = close.rolling(5).mean()
    data["MA10"] = close.rolling(10).mean()
    data["MA20"] = close.rolling(20).mean()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    data["EMA9"] = close.ewm(
        span=9,
        adjust=False
    ).mean()

    data["EMA20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    data["EMA50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    data["EMA100"] = close.ewm(
        span=100,
        adjust=False
    ).mean()

    data["EMA200"] = close.ewm(
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

    rs = avg_gain / avg_loss.replace(
        0,
        np.nan
    )

    data["RSI14"] = 100 - (
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

    data["MACD"] = ema12 - ema26

    data["MACD_SIGNAL"] = data["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    data["MACD_HIST"] = (
        data["MACD"] -
        data["MACD_SIGNAL"]
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    data["MOMENTUM"] = (
        close - close.shift(10)
    )

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

    data["ATR14"] = true_range.rolling(
        14
    ).mean()

    # --------------------------------------------------------
    # ADX / DI
    # --------------------------------------------------------

    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = pd.Series(
        np.where(
            (up_move > down_move) &
            (up_move > 0),
            up_move,
            0
        ),
        index=data.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) &
            (down_move > 0),
            down_move,
            0
        ),
        index=data.index
    )

    atr_for_adx = true_range.rolling(14).mean()

    plus_di = (
        100 *
        plus_dm.rolling(14).mean() /
        atr_for_adx.replace(0, np.nan)
    )

    minus_di = (
        100 *
        minus_dm.rolling(14).mean() /
        atr_for_adx.replace(0, np.nan)
    )

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(
            0,
            np.nan
        )
    )

    data["PLUS_DI"] = plus_di
    data["MINUS_DI"] = minus_di
    data["ADX14"] = dx.rolling(14).mean()

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    typical_price = (
        high + low + close
    ) / 3

    volume = data["Volume"]

    if (
        volume.notna().any() and
        volume.fillna(0).sum() > 0
    ):

        cumulative_volume = (
            volume.fillna(0).cumsum()
        )

        cumulative_pv = (
            typical_price *
            volume.fillna(0)
        ).cumsum()

        data["VWAP"] = (
            cumulative_pv /
            cumulative_volume.replace(
                0,
                np.nan
            )
        )

    else:

        data["VWAP"] = np.nan

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    data["BB_MIDDLE"] = bb_middle

    data["BB_UPPER"] = (
        bb_middle +
        2 * bb_std
    )

    data["BB_LOWER"] = (
        bb_middle -
        2 * bb_std
    )

    data["BB_WIDTH"] = (
        (
            data["BB_UPPER"] -
            data["BB_LOWER"]
        ) /
        bb_middle.replace(
            0,
            np.nan
        )
    )

    return data


# ============================================================
# SUPPORT / RESISTANCE
# ============================================================

def calculate_support_resistance(
    df: pd.DataFrame
):

    if len(df) < 20:

        return {
            "support": None,
            "support2": None,
            "resistance": None,
            "resistance2": None
        }

    reference = df.iloc[:-1]

    recent_20 = reference.tail(20)
    recent_50 = reference.tail(50)

    support = safe_float(
        recent_20["Low"].min()
    )

    resistance = safe_float(
        recent_20["High"].max()
    )

    support2 = safe_float(
        recent_50["Low"].min()
    )

    resistance2 = safe_float(
        recent_50["High"].max()
    )

    return {
        "support": support,
        "support2": support2,
        "resistance": resistance,
        "resistance2": resistance2
    }


# ============================================================
# PREVIOUS DAY LEVELS
# ============================================================

def get_previous_day_levels():

    try:

        df = fetch_history("1D")

        if df.empty:
            return {
                "high": None,
                "low": None,
                "close": None,
                "date": None
            }

        dates = pd.Series(
            df.index.date,
            index=df.index
        )

        today = now_ist().date()

        previous = df.loc[
            dates < today
        ]

        if previous.empty:

            previous = df.iloc[:-1]

        if previous.empty:

            return {
                "high": None,
                "low": None,
                "close": None,
                "date": None
            }

        row = previous.iloc[-1]

        previous_date = previous.index[-1]

        return {
            "high": safe_float(row["High"]),
            "low": safe_float(row["Low"]),
            "close": safe_float(row["Close"]),
            "date": previous_date.strftime("%Y-%m-%d")
        }

    except Exception:

        return {
            "high": None,
            "low": None,
            "close": None,
            "date": None
        }


# ============================================================
# DAY OPEN / HIGH / LOW
# ============================================================

def get_day_statistics(
    df: pd.DataFrame
):

    if df.empty:

        return {
            "open": None,
            "high": None,
            "low": None
        }

    try:

        today = now_ist().date()

        dates = pd.Series(
            df.index.date,
            index=df.index
        )

        today_data = df.loc[
            dates == today
        ]

        if today_data.empty:

            return {
                "open": None,
                "high": None,
                "low": None
            }

        return {
            "open": safe_float(
                today_data["Open"].iloc[0]
            ),
            "high": safe_float(
                today_data["High"].max()
            ),
            "low": safe_float(
                today_data["Low"].min()
            )
        }

    except Exception:

        return {
            "open": None,
            "high": None,
            "low": None
        }


# ============================================================
# PIVOTS
# ============================================================

def calculate_pivots(
    previous_high,
    previous_low,
    previous_close
):

    if (
        previous_high is None or
        previous_low is None or
        previous_close is None
    ):

        return {
            "pivot": None,
            "r1": None,
            "r2": None,
            "s1": None,
            "s2": None
        }

    try:

        h = float(previous_high)
        l = float(previous_low)
        c = float(previous_close)

        pivot = (
            h + l + c
        ) / 3

        r1 = (
            2 * pivot
        ) - l

        s1 = (
            2 * pivot
        ) - h

        r2 = (
            pivot + h - l
        )

        s2 = (
            pivot - h + l
        )

        return {
            "pivot": safe_float(pivot),
            "r1": safe_float(r1),
            "r2": safe_float(r2),
            "s1": safe_float(s1),
            "s2": safe_float(s2)
        }

    except Exception:

        return {
            "pivot": None,
            "r1": None,
            "r2": None,
            "s1": None,
            "s2": None
        }


# ============================================================
# VOLUME ANALYSIS
# ============================================================

def calculate_volume_status(
    df: pd.DataFrame
):

    if (
        "Volume" not in df.columns or
        len(df) < 21
    ):

        return {
            "status": "UNAVAILABLE",
            "ratio": None,
            "available": False
        }

    volume = df["Volume"]

    if (
        volume.isna().all() or
        volume.fillna(0).sum() <= 0
    ):

        return {
            "status": "UNAVAILABLE",
            "ratio": None,
            "available": False
        }

    current = volume.iloc[-1]

    average = volume.iloc[-21:-1].mean()

    if (
        average is None or
        not np.isfinite(average) or
        average <= 0
    ):

        return {
            "status": "UNAVAILABLE",
            "ratio": None,
            "available": False
        }

    ratio = current / average

    if ratio >= 1.5:
        status = "HIGH"

    elif ratio <= 0.7:
        status = "LOW"

    else:
        status = "NORMAL"

    return {
        "status": status,
        "ratio": safe_float(ratio, 2),
        "available": True
    }


# ============================================================
# MARKET STRUCTURE
# ============================================================

def calculate_market_structure(
    df: pd.DataFrame
):

    if len(df) < 12:
        return "UNKNOWN"

    recent = df.tail(10)

    first_half = recent.iloc[:5]
    second_half = recent.iloc[5:]

    first_high = first_half["High"].max()
    second_high = second_half["High"].max()

    first_low = first_half["Low"].min()
    second_low = second_half["Low"].min()

    if (
        second_high > first_high and
        second_low > first_low
    ):
        return "HH_HL"

    if (
        second_high < first_high and
        second_low < first_low
    ):
        return "LH_LL"

    first_range = (
        first_high - first_low
    )

    second_range = (
        second_high - second_low
    )

    if (
        first_range > 0 and
        second_range >
        first_range * 1.35
    ):
        return "EXPANSION"

    return "RANGE"


# ============================================================
# CANDLE PATTERN
# ============================================================

def detect_candle_pattern(
    df: pd.DataFrame
):

    if len(df) < 3:
        return "NONE"

    prev = df.iloc[-2]
    current = df.iloc[-1]

    prev_open = float(prev["Open"])
    prev_close = float(prev["Close"])

    current_open = float(current["Open"])
    current_close = float(current["Close"])

    current_high = float(current["High"])
    current_low = float(current["Low"])

    current_body = abs(
        current_close -
        current_open
    )

    current_range = (
        current_high -
        current_low
    )

    if current_range <= 0:
        return "NONE"

    # --------------------------------------------------------
    # Bullish engulfing
    # --------------------------------------------------------

    if (
        prev_close < prev_open and
        current_close > current_open and
        current_open <= prev_close and
        current_close >= prev_open
    ):
        return "BULLISH_ENGULFING"

    # --------------------------------------------------------
    # Bearish engulfing
    # --------------------------------------------------------

    if (
        prev_close > prev_open and
        current_close < current_open and
        current_open >= prev_close and
        current_close <= prev_open
    ):
        return "BEARISH_ENGULFING"

    upper_wick = (
        current_high -
        max(current_open, current_close)
    )

    lower_wick = (
        min(current_open, current_close) -
        current_low
    )

    # --------------------------------------------------------
    # Hammer
    # --------------------------------------------------------

    if (
        lower_wick >= current_body * 2 and
        upper_wick <= current_body
    ):
        return "HAMMER"

    # --------------------------------------------------------
    # Shooting star
    # --------------------------------------------------------

    if (
        upper_wick >= current_body * 2 and
        lower_wick <= current_body
    ):
        return "SHOOTING_STAR"

    # --------------------------------------------------------
    # Strong candle
    # --------------------------------------------------------

    if (
        current_body >=
        current_range * 0.70
    ):

        if current_close > current_open:
            return "STRONG_BULLISH"

        if current_close < current_open:
            return "STRONG_BEARISH"

    return "NONE"


# ============================================================
# VOLATILITY
# ============================================================

def calculate_volatility(
    df: pd.DataFrame
):

    if len(df) < 35:

        return {
            "status": "UNKNOWN",
            "atr_percent": None,
            "ratio": None
        }

    atr = df["ATR14"].iloc[-1]
    close = df["Close"].iloc[-1]

    median_atr = (
        df["ATR14"]
        .iloc[-31:-1]
        .median()
    )

    if (
        atr is None or
        close is None or
        median_atr is None or
        not np.isfinite(atr) or
        not np.isfinite(close) or
        not np.isfinite(median_atr) or
        median_atr <= 0
    ):

        return {
            "status": "UNKNOWN",
            "atr_percent": None,
            "ratio": None
        }

    ratio = atr / median_atr

    atr_percent = (
        atr / close
    ) * 100

    if ratio >= 1.35:
        status = "HIGH"

    elif ratio <= 0.70:
        status = "LOW"

    else:
        status = "NORMAL"

    return {
        "status": status,
        "atr_percent": safe_float(
            atr_percent,
            3
        ),
        "ratio": safe_float(
            ratio,
            2
        )
    }


# ============================================================
# BOLLINGER STATE
# ============================================================

def calculate_bollinger_state(
    df: pd.DataFrame
):

    if len(df) < 30:

        return {
            "squeeze": False,
            "expansion": False,
            "state": "UNKNOWN"
        }

    current_width = df["BB_WIDTH"].iloc[-1]

    median_width = (
        df["BB_WIDTH"]
        .iloc[-21:-1]
        .median()
    )

    previous_width = (
        df["BB_WIDTH"].iloc[-2]
    )

    if (
        current_width is None or
        median_width is None or
        not np.isfinite(current_width) or
        not np.isfinite(median_width) or
        median_width <= 0
    ):

        return {
            "squeeze": False,
            "expansion": False,
            "state": "UNKNOWN"
        }

    squeeze = (
        current_width <=
        median_width * 0.75
    )

    expansion = False

    if (
        previous_width is not None and
        np.isfinite(previous_width) and
        previous_width > 0
    ):

        expansion = (
            current_width >=
            previous_width * 1.15
        )

    if expansion:
        state = "EXPANSION"

    elif squeeze:
        state = "SQUEEZE"

    else:
        state = "NORMAL"

    return {
        "squeeze": bool(squeeze),
        "expansion": bool(expansion),
        "state": state
    }


# ============================================================
# TREND
# ============================================================

def calculate_trend(
    df: pd.DataFrame
):

    if len(df) < 50:
        return "UNKNOWN"

    row = df.iloc[-1]

    close = row["Close"]
    ema9 = row["EMA9"]
    ema20 = row["EMA20"]
    ema50 = row["EMA50"]

    if any(
        pd.isna(x)
        for x in [
            close,
            ema9,
            ema20,
            ema50
        ]
    ):
        return "UNKNOWN"

    if (
        close > ema20 and
        ema9 > ema20 and
        ema20 > ema50
    ):
        return "BULLISH"

    if (
        close < ema20 and
        ema9 < ema20 and
        ema20 < ema50
    ):
        return "BEARISH"

    return "SIDEWAYS"


# ============================================================
# LONGER TREND USING EMA200
# ============================================================

def calculate_long_trend(
    df: pd.DataFrame
):

    if len(df) < 100:
        return "UNKNOWN"

    row = df.iloc[-1]

    close = row["Close"]
    ema50 = row["EMA50"]
    ema100 = row["EMA100"]
    ema200 = row["EMA200"]

    if any(
        pd.isna(x)
        for x in [
            close,
            ema50,
            ema100,
            ema200
        ]
    ):
        return "UNKNOWN"

    if (
        close > ema50 and
        ema50 > ema100 and
        ema100 > ema200
    ):
        return "STRONG_BULLISH"

    if (
        close < ema50 and
        ema50 < ema100 and
        ema100 < ema200
    ):
        return "STRONG_BEARISH"

    return "MIXED"


# ============================================================
# MARKET REGIME
# ============================================================

def calculate_market_regime(
    df: pd.DataFrame
):

    if len(df) < 50:
        return "UNKNOWN"

    row = df.iloc[-1]

    adx = row["ADX14"]
    plus_di = row["PLUS_DI"]
    minus_di = row["MINUS_DI"]

    if any(
        pd.isna(x)
        for x in [
            adx,
            plus_di,
            minus_di
        ]
    ):
        return "UNKNOWN"

    if adx >= 25:

        if plus_di > minus_di:
            return "TREND_UP"

        if minus_di > plus_di:
            return "TREND_DOWN"

    return "RANGE_CHOP"


# ============================================================
# BREAKOUT / BREAKDOWN
# ============================================================

def detect_breakout(
    df: pd.DataFrame,
    sr: dict,
    volume_info: dict
):

    if len(df) < 20:

        return {
            "breakout_confirmed": False,
            "breakdown_confirmed": False,
            "breakout_level": None,
            "breakdown_level": None,
            "status": "NONE",
            "strength": "NONE"
        }

    current = df.iloc[-1]

    close = float(current["Close"])
    high = float(current["High"])
    low = float(current["Low"])

    resistance = sr.get("resistance")
    support = sr.get("support")

    breakout = False
    breakdown = False

    breakout_level = None
    breakdown_level = None

    if resistance is not None:

        if close > resistance:
            breakout = True
            breakout_level = resistance

    if support is not None:

        if close < support:
            breakdown = True
            breakdown_level = support

    volume_available = volume_info.get(
        "available",
        False
    )

    volume_ratio = volume_info.get(
        "ratio"
    )

    volume_confirmed = False

    if (
        volume_available and
        volume_ratio is not None and
        volume_ratio >= 1.20
    ):
        volume_confirmed = True

    if breakout:

        if volume_confirmed:
            strength = "STRONG"

        elif not volume_available:
            strength = "PRICE_CONFIRMED"

        else:
            strength = "WEAK"

        return {
            "breakout_confirmed": True,
            "breakdown_confirmed": False,
            "breakout_level": safe_float(
                breakout_level
            ),
            "breakdown_level": None,
            "status": "BREAKOUT",
            "strength": strength
        }

    if breakdown:

        if volume_confirmed:
            strength = "STRONG"

        elif not volume_available:
            strength = "PRICE_CONFIRMED"

        else:
            strength = "WEAK"

        return {
            "breakout_confirmed": False,
            "breakdown_confirmed": True,
            "breakout_level": None,
            "breakdown_level": safe_float(
                breakdown_level
            ),
            "status": "BREAKDOWN",
            "strength": strength
        }

    return {
        "breakout_confirmed": False,
        "breakdown_confirmed": False,
        "breakout_level": None,
        "breakdown_level": None,
        "status": "NONE",
        "strength": "NONE"
    }


# ============================================================
# FALSE BREAKOUT / BREAKDOWN
# ============================================================

def detect_false_breakout(
    df: pd.DataFrame,
    sr: dict
):

    if len(df) < 20:

        return {
            "false_breakout": False,
            "false_breakdown": False,
            "status": "NONE"
        }

    current = df.iloc[-1]

    close = float(current["Close"])
    high = float(current["High"])
    low = float(current["Low"])

    resistance = sr.get("resistance")
    support = sr.get("support")

    false_breakout = False
    false_breakdown = False

    if resistance is not None:

        false_breakout = (
            high > resistance and
            close <= resistance
        )

    if support is not None:

        false_breakdown = (
            low < support and
            close >= support
        )

    if false_breakout:

        status = "FALSE_BREAKOUT"

    elif false_breakdown:

        status = "FALSE_BREAKDOWN"

    else:

        status = "NONE"

    return {
        "false_breakout": bool(false_breakout),
        "false_breakdown": bool(false_breakdown),
        "status": status
    }


# ============================================================
# HIGHER TIMEFRAME ANALYSIS
# ============================================================

def get_higher_timeframe_analysis():

    intervals = [
        "15m",
        "30m",
        "1h",
        "1D"
    ]

    result = {}

    for interval in intervals:

        try:

            df = fetch_history(interval)

            if df.empty:

                result[interval] = {
                    "trend": "UNKNOWN",
                    "long_trend": "UNKNOWN"
                }

                continue

            df = calculate_indicators(df)

            result[interval] = {
                "trend": calculate_trend(df),
                "long_trend": calculate_long_trend(df),
                "last_price": safe_float(
                    df["Close"].iloc[-1]
                )
            }

        except Exception:

            result[interval] = {
                "trend": "UNKNOWN",
                "long_trend": "UNKNOWN",
                "last_price": None
            }

    return result


# ============================================================
# MULTI-TIMEFRAME ALIGNMENT
# ============================================================

def calculate_mtf_alignment(
    timeframe_data: dict
):

    trends = []

    for interval in [
        "15m",
        "30m",
        "1h",
        "1D"
    ]:

        trend = (
            timeframe_data
            .get(interval, {})
            .get("trend")
        )

        if trend in [
            "BULLISH",
            "BEARISH"
        ]:

            trends.append(trend)

    if len(trends) < 2:
        return "INSUFFICIENT_DATA"

    bullish = trends.count("BULLISH")
    bearish = trends.count("BEARISH")

    if bullish == len(trends):
        return "FULL_BULLISH"

    if bearish == len(trends):
        return "FULL_BEARISH"

    if bullish >= len(trends) * 0.75:
        return "BULLISH_BIAS"

    if bearish >= len(trends) * 0.75:
        return "BEARISH_BIAS"

    return "MIXED"


# ============================================================
# HTF CONFLICT
# ============================================================

def detect_htf_conflict(
    timeframe_data: dict
):

    trends = []

    for interval in [
        "15m",
        "30m",
        "1h",
        "1D"
    ]:

        trend = (
            timeframe_data
            .get(interval, {})
            .get("trend")
        )

        if trend in [
            "BULLISH",
            "BEARISH"
        ]:
            trends.append(trend)

    if not trends:
        return False

    bullish = trends.count("BULLISH")
    bearish = trends.count("BEARISH")

    return (
        bullish > 0 and
        bearish > 0
    )


# ============================================================
# SIGNAL ENGINE
# ============================================================

def generate_signal(
    df: pd.DataFrame,
    market_status: str,
    volume_info: dict,
    structure: str,
    candle: str,
    volatility: dict,
    bollinger: dict,
    breakout: dict,
    timeframe_data: dict
):

    row = df.iloc[-1]

    close = row["Close"]

    bullish = 0.0
    bearish = 0.0

    bullish_reasons = []
    bearish_reasons = []

    # ========================================================
    # EMA 9 / 20
    # ========================================================

    if (
        row["EMA9"] >
        row["EMA20"]
    ):

        bullish += 1
        bullish_reasons.append(
            "EMA9 above EMA20"
        )

    elif (
        row["EMA9"] <
        row["EMA20"]
    ):

        bearish += 1
        bearish_reasons.append(
            "EMA9 below EMA20"
        )

    # ========================================================
    # EMA 20 / 50
    # ========================================================

    if (
        row["EMA20"] >
        row["EMA50"]
    ):

        bullish += 1
        bullish_reasons.append(
            "EMA20 above EMA50"
        )

    elif (
        row["EMA20"] <
        row["EMA50"]
    ):

        bearish += 1
        bearish_reasons.append(
            "EMA20 below EMA50"
        )

    # ========================================================
    # EMA 50 / 100 / 200
    # ========================================================

    if (
        row["EMA50"] >
        row["EMA100"] >
        row["EMA200"]
    ):

        bullish += 1.0

        bullish_reasons.append(
            "Long-term EMA structure bullish"
        )

    elif (
        row["EMA50"] <
        row["EMA100"] <
        row["EMA200"]
    ):

        bearish += 1.0

        bearish_reasons.append(
            "Long-term EMA structure bearish"
        )

    # ========================================================
    # PRICE vs EMA20
    # ========================================================

    if close > row["EMA20"]:

        bullish += 1
        bullish_reasons.append(
            "Price above EMA20"
        )

    elif close < row["EMA20"]:

        bearish += 1
        bearish_reasons.append(
            "Price below EMA20"
        )

    # ========================================================
    # RSI
    # ========================================================

    rsi = row["RSI14"]

    if not pd.isna(rsi):

        if 55 <= rsi <= 70:

            bullish += 1
            bullish_reasons.append(
                "RSI bullish"
            )

        elif 30 <= rsi <= 45:

            bearish += 1
            bearish_reasons.append(
                "RSI bearish"
            )

        elif rsi > 70:

            bearish += 0.5

        elif rsi < 30:

            bullish += 0.5

    # ========================================================
    # MACD
    # ========================================================

    macd = row["MACD"]
    macd_signal = row["MACD_SIGNAL"]
    macd_hist = row["MACD_HIST"]

    if (
        not pd.isna(macd) and
        not pd.isna(macd_signal)
    ):

        if macd > macd_signal:

            bullish += 1
            bullish_reasons.append(
                "MACD bullish"
            )

        elif macd < macd_signal:

            bearish += 1
            bearish_reasons.append(
                "MACD bearish"
            )

    if not pd.isna(macd_hist):

        if macd_hist > 0:
            bullish += 0.5

        elif macd_hist < 0:
            bearish += 0.5

    # ========================================================
    # VWAP
    # ========================================================

    vwap = row["VWAP"]

    if not pd.isna(vwap):

        if close > vwap:

            bullish += 1
            bullish_reasons.append(
                "Price above VWAP"
            )

        elif close < vwap:

            bearish += 1
            bearish_reasons.append(
                "Price below VWAP"
            )

    # ========================================================
    # MOMENTUM
    # ========================================================

    momentum = row["MOMENTUM"]

    if not pd.isna(momentum):

        if momentum > 0:

            bullish += 1
            bullish_reasons.append(
                "Positive momentum"
            )

        elif momentum < 0:

            bearish += 1
            bearish_reasons.append(
                "Negative momentum"
            )

    # ========================================================
    # ADX + DI
    # ========================================================

    adx = row["ADX14"]
    plus_di = row["PLUS_DI"]
    minus_di = row["MINUS_DI"]

    if not any(
        pd.isna(x)
        for x in [
            adx,
            plus_di,
            minus_di
        ]
    ):

        if adx >= 20:

            if plus_di > minus_di:

                bullish += 1
                bullish_reasons.append(
                    "Directional strength favors buyers"
                )

            elif minus_di > plus_di:

                bearish += 1
                bearish_reasons.append(
                    "Directional strength favors sellers"
                )

    # ========================================================
    # MARKET STRUCTURE
    # ========================================================

    if structure == "HH_HL":

        bullish += 1
        bullish_reasons.append(
            "Higher-high / higher-low structure"
        )

    elif structure == "LH_LL":

        bearish += 1
        bearish_reasons.append(
            "Lower-high / lower-low structure"
        )

    elif structure == "RANGE":

        bullish -= 0.25
        bearish -= 0.25

    # ========================================================
    # CANDLE
    # ========================================================

    if candle in [
        "BULLISH_ENGULFING",
        "HAMMER",
        "STRONG_BULLISH"
    ]:

        bullish += 1
        bullish_reasons.append(
            f"{candle.replace('_', ' ').title()} candle"
        )

    elif candle in [
        "BEARISH_ENGULFING",
        "SHOOTING_STAR",
        "STRONG_BEARISH"
    ]:

        bearish += 1
        bearish_reasons.append(
            f"{candle.replace('_', ' ').title()} candle"
        )

    # ========================================================
    # BREAKOUT
    # ========================================================

    if breakout["breakout_confirmed"]:

        bullish += 2

        bullish_reasons.append(
            "Resistance breakout"
        )

        if breakout["strength"] == "STRONG":
            bullish += 0.5

    if breakout["breakdown_confirmed"]:

        bearish += 2

        bearish_reasons.append(
            "Support breakdown"
        )

        if breakout["strength"] == "STRONG":
            bearish += 0.5

    # ========================================================
    # VOLUME
    # ========================================================

    if volume_info["available"]:

        ratio = volume_info["ratio"]

        if (
            ratio is not None and
            ratio >= 1.20
        ):

            if bullish > bearish:

                bullish += 1
                bullish_reasons.append(
                    "Volume confirmation"
                )

            elif bearish > bullish:

                bearish += 1
                bearish_reasons.append(
                    "Volume confirmation"
                )

    # ========================================================
    # MTF
    # ========================================================

    alignment = calculate_mtf_alignment(
        timeframe_data
    )

    if alignment == "FULL_BULLISH":

        bullish += 2

        bullish_reasons.append(
            "Higher timeframes aligned bullish"
        )

    elif alignment == "BULLISH_BIAS":

        bullish += 1

    elif alignment == "FULL_BEARISH":

        bearish += 2

        bearish_reasons.append(
            "Higher timeframes aligned bearish"
        )

    elif alignment == "BEARISH_BIAS":

        bearish += 1

    # ========================================================
    # MARKET REGIME
    # ========================================================

    regime = calculate_market_regime(df)

    if regime == "TREND_UP":

        bullish += 1

        bullish_reasons.append(
            "Trending market favors buyers"
        )

    elif regime == "TREND_DOWN":

        bearish += 1

        bearish_reasons.append(
            "Trending market favors sellers"
        )

    elif regime == "RANGE_CHOP":

        bullish -= 0.5
        bearish -= 0.5

    # ========================================================
    # VOLATILITY FILTER
    # ========================================================

    volatility_status = volatility.get(
        "status"
    )

    if volatility_status == "LOW":

        bullish -= 0.5
        bearish -= 0.5

    # ========================================================
    # BOLLINGER EXPANSION
    # ========================================================

    if bollinger.get("expansion"):

        if bullish > bearish:
            bullish += 0.5

        elif bearish > bullish:
            bearish += 0.5

    # ========================================================
    # FINAL SCORE
    # ========================================================

    bullish = max(
        0.0,
        bullish
    )

    bearish = max(
        0.0,
        bearish
    )

    net_score = (
        bullish -
        bearish
    )

    dominant = max(
        bullish,
        bearish
    )

    total = (
        bullish +
        bearish
    )

    if total > 0:

        confidence = (
            dominant /
            total
        ) * 100

    else:

        confidence = 0

    confidence = min(
        confidence,
        95
    )

    # ========================================================
    # HTF CONFLICT
    # ========================================================

    htf_conflict = detect_htf_conflict(
        timeframe_data
    )

    # ========================================================
    # DECISION
    # ========================================================

    decision = "WAIT"

    no_trade_reason = ""

    if market_status not in [
        "LIVE",
        "DELAYED"
    ]:

        no_trade_reason = (
            "Market is not in active trading state."
        )

    elif htf_conflict:

        no_trade_reason = (
            "Higher timeframes are conflicting."
        )

    elif regime == "RANGE_CHOP":

        no_trade_reason = (
            "Market is range-bound/choppy."
        )

    elif volatility_status == "LOW":

        no_trade_reason = (
            "Volatility is too low for a strong setup."
        )

    elif (
        bullish >= 10 and
        net_score >= 3 and
        confidence >= 58
    ):

        decision = "BUY"

    elif (
        bearish >= 10 and
        net_score <= -3 and
        confidence >= 58
    ):

        decision = "SELL"

    else:

        no_trade_reason = (
            "Signal strength is not strong enough."
        )

    # --------------------------------------------------------
    # Compatibility field
    # --------------------------------------------------------

    if decision == "BUY":
        signal = "BUY"

    elif decision == "SELL":
        signal = "SELL"

    else:
        signal = "NEUTRAL"

    # --------------------------------------------------------
    # Signal quality
    # --------------------------------------------------------

    if decision in [
        "BUY",
        "SELL"
    ]:

        if confidence >= 75:
            quality = "HIGH"

        elif confidence >= 62:
            quality = "MEDIUM"

        else:
            quality = "LOW"

    else:

        quality = "LOW"

    # --------------------------------------------------------
    # Strength
    # --------------------------------------------------------

    strength = int(
        round(confidence)
    )

    return {
        "signal": signal,
        "decision": decision,
        "confidence": safe_float(
            confidence,
            1
        ),
        "strength": strength,
        "bullish_score": safe_float(
            bullish,
            1
        ),
        "bearish_score": safe_float(
            bearish,
            1
        ),
        "net_score": safe_float(
            net_score,
            1
        ),
        "quality": quality,
        "bullish_reasons": bullish_reasons[
            :8
        ],
        "bearish_reasons": bearish_reasons[
            :8
        ],
        "no_trade_reason": no_trade_reason,
        "market_regime": regime,
        "htf_conflict": htf_conflict,
        "mtf_alignment": alignment
    }


# ============================================================
# TRADE PLAN
# ============================================================

def calculate_trade_plan(
    df: pd.DataFrame,
    signal_info: dict,
    sr: dict
):

    decision = signal_info.get(
        "decision"
    )

    if decision not in [
        "BUY",
        "SELL"
    ]:

        return {
            "status": "NO TRADE",
            "entry": None,
            "entry_low": None,
            "entry_high": None,
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "trailing_stop": None,
            "risk_points": None,
            "reward_1_points": None,
            "reward_2_points": None,
            "risk_reward_1": None,
            "risk_reward_2": None
        }

    row = df.iloc[-1]

    entry = float(row["Close"])
    atr = row["ATR14"]

    if (
        atr is None or
        pd.isna(atr) or
        atr <= 0
    ):

        return {
            "status": "NO TRADE",
            "entry": None,
            "entry_low": None,
            "entry_high": None,
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "trailing_stop": None,
            "risk_points": None,
            "reward_1_points": None,
            "reward_2_points": None,
            "risk_reward_1": None,
            "risk_reward_2": None
        }

    atr = float(atr)

    # ========================================================
    # BUY
    # ========================================================

    if decision == "BUY":

        base_sl = (
            entry -
            atr * 1.2
        )

        support = sr.get("support")

        if support is not None:

            support_sl = (
                support -
                atr * 0.20
            )

            stop_loss = min(
                base_sl,
                support_sl
            )

        else:

            stop_loss = base_sl

        risk = (
            entry -
            stop_loss
        )

        target_1 = (
            entry +
            max(
                atr * 1.8,
                risk * 1.5
            )
        )

        target_2 = (
            entry +
            max(
                atr * 2.8,
                risk * 2.3
            )
        )

        trailing_stop = (
            entry -
            atr
        )

    # ========================================================
    # SELL
    # ========================================================

    else:

        base_sl = (
            entry +
            atr * 1.2
        )

        resistance = sr.get(
            "resistance"
        )

        if resistance is not None:

            resistance_sl = (
                resistance +
                atr * 0.20
            )

            stop_loss = max(
                base_sl,
                resistance_sl
            )

        else:

            stop_loss = base_sl

        risk = (
            stop_loss -
            entry
        )

        target_1 = (
            entry -
            max(
                atr * 1.8,
                risk * 1.5
            )
        )

        target_2 = (
            entry -
            max(
                atr * 2.8,
                risk * 2.3
            )
        )

        trailing_stop = (
            entry +
            atr
        )

    if risk <= 0:

        return {
            "status": "NO TRADE",
            "entry": None,
            "entry_low": None,
            "entry_high": None,
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "trailing_stop": None,
            "risk_points": None,
            "reward_1_points": None,
            "reward_2_points": None,
            "risk_reward_1": None,
            "risk_reward_2": None
        }

    reward1 = abs(
        target_1 -
        entry
    )

    reward2 = abs(
        target_2 -
        entry
    )

    rr1 = reward1 / risk
    rr2 = reward2 / risk

    entry_buffer = atr * 0.25

    entry_low = (
        entry -
        entry_buffer
    )

    entry_high = (
        entry +
        entry_buffer
    )

    return {
        "status": "TRADE",
        "entry": safe_float(entry),
        "entry_low": safe_float(
            entry_low
        ),
        "entry_high": safe_float(
            entry_high
        ),
        "stop_loss": safe_float(
            stop_loss
        ),
        "target_1": safe_float(
            target_1
        ),
        "target_2": safe_float(
            target_2
        ),
        "trailing_stop": safe_float(
            trailing_stop
        ),
        "risk_points": safe_float(
            risk
        ),
        "reward_1_points": safe_float(
            reward1
        ),
        "reward_2_points": safe_float(
            reward2
        ),
        "risk_reward_1": safe_float(
            rr1,
            2
        ),
        "risk_reward_2": safe_float(
            rr2,
            2
        )
    }


# ============================================================
# ENTRY / AVOID ZONES
# ============================================================

def calculate_zones(
    df: pd.DataFrame,
    trade_plan: dict,
    signal_info: dict,
    sr: dict
):

    if trade_plan["status"] != "TRADE":

        return {
            "entry_zone": None,
            "avoid_zone": (
                signal_info.get(
                    "no_trade_reason"
                ) or
                "No high-quality setup."
            )
        }

    entry_low = trade_plan.get(
        "entry_low"
    )

    entry_high = trade_plan.get(
        "entry_high"
    )

    decision = signal_info.get(
        "decision"
    )

    if (
        entry_low is None or
        entry_high is None
    ):

        entry_zone = None

    else:

        entry_zone = (
            f"{entry_low:.2f} - "
            f"{entry_high:.2f}"
        )

    if decision == "BUY":

        resistance = sr.get(
            "resistance"
        )

        if resistance is not None:

            avoid_zone = (
                f"Avoid fresh BUY near "
                f"{resistance:.2f} resistance "
                f"unless breakout holds."
            )

        else:

            avoid_zone = (
                "Avoid chasing extended candles."
            )

    else:

        support = sr.get(
            "support"
        )

        if support is not None:

            avoid_zone = (
                f"Avoid fresh SELL near "
                f"{support:.2f} support "
                f"unless breakdown holds."
            )

        else:

            avoid_zone = (
                "Avoid chasing extended candles."
            )

    return {
        "entry_zone": entry_zone,
        "avoid_zone": avoid_zone
    }


# ============================================================
# HUMAN-READABLE AI ANALYSIS
# ============================================================

def build_analysis_text(
    signal_info: dict,
    trade_plan: dict,
    trend: str,
    regime: str,
    structure: str,
    mtf_alignment: str,
    breakout: dict,
    volatility: dict
):

    decision = signal_info.get(
        "decision"
    )

    confidence = signal_info.get(
        "confidence"
    )

    if decision == "BUY":

        text = (
            f"BUY bias detected with "
            f"{confidence}% confidence. "
        )

        if trend == "BULLISH":
            text += "Short-term trend is bullish. "

        if structure == "HH_HL":
            text += "Market structure is making higher highs and higher lows. "

        if mtf_alignment in [
            "FULL_BULLISH",
            "BULLISH_BIAS"
        ]:
            text += "Higher timeframes support the bullish bias. "

        if breakout.get(
            "breakout_confirmed"
        ):
            text += "Resistance breakout is detected. "

        if volatility.get(
            "status"
        ) == "HIGH":
            text += "Volatility is elevated, so risk management is important. "

        if trade_plan.get(
            "status"
        ) == "TRADE":

            text += (
                f"Entry around "
                f"{trade_plan['entry']:.2f}, "
                f"SL {trade_plan['stop_loss']:.2f}, "
                f"T1 {trade_plan['target_1']:.2f}, "
                f"T2 {trade_plan['target_2']:.2f}."
            )

        return text

    if decision == "SELL":

        text = (
            f"SELL bias detected with "
            f"{confidence}% confidence. "
        )

        if trend == "BEARISH":
            text += "Short-term trend is bearish. "

        if structure == "LH_LL":
            text += "Market structure is making lower highs and lower lows. "

        if mtf_alignment in [
            "FULL_BEARISH",
            "BEARISH_BIAS"
        ]:
            text += "Higher timeframes support the bearish bias. "

        if breakout.get(
            "breakdown_confirmed"
        ):
            text += "Support breakdown is detected. "

        if volatility.get(
            "status"
        ) == "HIGH":
            text += "Volatility is elevated, so risk management is important. "

        if trade_plan.get(
            "status"
        ) == "TRADE":

            text += (
                f"Entry around "
                f"{trade_plan['entry']:.2f}, "
                f"SL {trade_plan['stop_loss']:.2f}, "
                f"T1 {trade_plan['target_1']:.2f}, "
                f"T2 {trade_plan['target_2']:.2f}."
            )

        return text

    reason = signal_info.get(
        "no_trade_reason"
    )

    if not reason:
        reason = (
            "No strong directional edge."
        )

    return (
        f"WAIT — {reason} "
        f"Current market regime: {regime}. "
        f"MTF alignment: {mtf_alignment}. "
        f"Best approach is to wait for stronger confirmation."
    )


# ============================================================
# MAIN ANALYSIS
# ============================================================

def build_analysis(
    interval: str = "5m"
):

    # ========================================================
    # MAIN DATA
    # ========================================================

    df = fetch_history(interval)

    if df.empty:

        raise HTTPException(
            status_code=503,
            detail="Market data unavailable."
        )

    df = calculate_indicators(
        df
    )

    current = df.iloc[-1]

    price = safe_float(
        current["Close"]
    )

    # ========================================================
    # TIMESTAMP
    # ========================================================

    latest_timestamp = df.index[-1]

    market_info = get_market_status(
        latest_timestamp
    )

    market_status = market_info[
        "status"
    ]

    try:

        data_age = (
            now_ist() -
            latest_timestamp
        ).total_seconds()

    except Exception:

        data_age = None

    # ========================================================
    # PREVIOUS CLOSE
    # ========================================================

    if len(df) >= 2:

        previous_close = safe_float(
            df["Close"].iloc[-2]
        )

    else:

        previous_close = None

    if (
        price is not None and
        previous_close is not None
    ):

        change = (
            price -
            previous_close
        )

        change_percent = (
            change /
            previous_close
        ) * 100

    else:

        change = None
        change_percent = None

    # ========================================================
    # DAY STATS
    # ========================================================

    day_stats = get_day_statistics(
        df
    )

    # ========================================================
    # INDICATORS
    # ========================================================

    indicators = {
        "ma5": safe_float(
            current["MA5"]
        ),
        "ma10": safe_float(
            current["MA10"]
        ),
        "ma20": safe_float(
            current["MA20"]
        ),
        "ema9": safe_float(
            current["EMA9"]
        ),
        "ema20": safe_float(
            current["EMA20"]
        ),
        "ema50": safe_float(
            current["EMA50"]
        ),
        "ema100": safe_float(
            current["EMA100"]
        ),
        "ema200": safe_float(
            current["EMA200"]
        ),
        "rsi": safe_float(
            current["RSI14"]
        ),
        "macd": safe_float(
            current["MACD"],
            4
        ),
        "macd_signal": safe_float(
            current["MACD_SIGNAL"],
            4
        ),
        "macd_histogram": safe_float(
            current["MACD_HIST"],
            4
        ),
        "momentum": safe_float(
            current["MOMENTUM"]
        ),
        "atr": safe_float(
            current["ATR14"]
        ),
        "adx": safe_float(
            current["ADX14"]
        ),
        "plus_di": safe_float(
            current["PLUS_DI"]
        ),
        "minus_di": safe_float(
            current["MINUS_DI"]
        ),
        "vwap": safe_float(
            current["VWAP"]
        ),
        "bollinger_middle": safe_float(
            current["BB_MIDDLE"]
        ),
        "bollinger_upper": safe_float(
            current["BB_UPPER"]
        ),
        "bollinger_lower": safe_float(
            current["BB_LOWER"]
        ),
        "bollinger_width": safe_float(
            current["BB_WIDTH"],
            4
        )
    }

    # ========================================================
    # SUPPORT / RESISTANCE
    # ========================================================

    sr = calculate_support_resistance(
        df
    )

    # ========================================================
    # VOLUME
    # ========================================================

    volume_info = calculate_volume_status(
        df
    )

    # ========================================================
    # STRUCTURE
    # ========================================================

    structure = calculate_market_structure(
        df
    )

    # ========================================================
    # CANDLE
    # ========================================================

    candle = detect_candle_pattern(
        df
    )

    # ========================================================
    # VOLATILITY
    # ========================================================

    volatility = calculate_volatility(
        df
    )

    # ========================================================
    # BOLLINGER
    # ========================================================

    bollinger = calculate_bollinger_state(
        df
    )

    # ========================================================
    # TREND
    # ========================================================

    trend = calculate_trend(
        df
    )

    long_trend = calculate_long_trend(
        df
    )

    regime = calculate_market_regime(
        df
    )

    # ========================================================
    # PREVIOUS DAY
    # ========================================================

    previous_day = get_previous_day_levels()

    # ========================================================
    # PIVOTS
    # ========================================================

    pivots = calculate_pivots(
        previous_day["high"],
        previous_day["low"],
        previous_day["close"]
    )

    # ========================================================
    # HIGHER TIMEFRAMES
    # ========================================================

    timeframe_data = (
        get_higher_timeframe_analysis()
    )

    mtf_alignment = calculate_mtf_alignment(
        timeframe_data
    )

    htf_conflict = detect_htf_conflict(
        timeframe_data
    )

    # ========================================================
    # BREAKOUT
    # ========================================================

    breakout = detect_breakout(
        df,
        sr,
        volume_info
    )

    # ========================================================
    # FALSE BREAKOUT
    # ========================================================

    false_breakout = detect_false_breakout(
        df,
        sr
    )

    # ========================================================
    # SIGNAL
    # ========================================================

    signal_info = generate_signal(
        df=df,
        market_status=market_status,
        volume_info=volume_info,
        structure=structure,
        candle=candle,
        volatility=volatility,
        bollinger=bollinger,
        breakout=breakout,
        timeframe_data=timeframe_data
    )

    # ========================================================
    # TRADE PLAN
    # ========================================================

    trade_plan = calculate_trade_plan(
        df,
        signal_info,
        sr
    )

    # ========================================================
    # ZONES
    # ========================================================

    zones = calculate_zones(
        df,
        trade_plan,
        signal_info,
        sr
    )

    # ========================================================
    # ANALYSIS TEXT
    # ========================================================

    analysis_text = build_analysis_text(
        signal_info=signal_info,
        trade_plan=trade_plan,
        trend=trend,
        regime=regime,
        structure=structure,
        mtf_alignment=mtf_alignment,
        breakout=breakout,
        volatility=volatility
    )

    # ========================================================
    # FLAT COMPATIBILITY FIELDS
    # ========================================================

    decision = signal_info[
        "decision"
    ]

    signal = signal_info[
        "signal"
    ]

    confidence = signal_info[
        "confidence"
    ]

    # ========================================================
    # DISTANCES
    # ========================================================

    support_distance_percent = None
    resistance_distance_percent = None

    if (
        price is not None and
        sr["support"] is not None and
        price != 0
    ):

        support_distance_percent = (
            (
                price -
                sr["support"]
            ) /
            price
        ) * 100

    if (
        price is not None and
        sr["resistance"] is not None and
        price != 0
    ):

        resistance_distance_percent = (
            (
                sr["resistance"] -
                price
            ) /
            price
        ) * 100

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    response = {

        # ----------------------------------------------------
        # SYSTEM
        # ----------------------------------------------------

        "app": "NIFTY AI TRADER",

        "analysis_version": "FINAL-1.0",

        "version": "FINAL-1.0",

        "data_source": (
            "Yahoo Finance (^NSEI)"
        ),

        "symbol": "NIFTY 50",

        "symbol_code": SYMBOL,

        "status": "ok",

        # ----------------------------------------------------
        # PRICE
        # ----------------------------------------------------

        "price": price,

        "previous_close": previous_close,

        "change": safe_float(
            change
        ),

        "change_percent": safe_float(
            change_percent,
            3
        ),

        "day_open": day_stats["open"],

        "day_high": day_stats["high"],

        "day_low": day_stats["low"],

        # ----------------------------------------------------
        # MARKET STATUS
        # ----------------------------------------------------

        "market_status": market_status,

        "market_message": market_info[
            "message"
        ],

        "data_age_seconds": safe_float(
            data_age,
            1
        ),

        "latest_data_time": (
            latest_timestamp.isoformat()
        ),

        # ----------------------------------------------------
        # MAIN DECISION
        # ----------------------------------------------------

        "signal": signal,

        "decision": decision,

        "decision_label": (
            "BUY" if decision == "BUY"
            else
            "SELL" if decision == "SELL"
            else
            "WAIT"
        ),

        "signal_strength": signal_info[
            "strength"
        ],

        "confidence": confidence,

        "signal_quality": signal_info[
            "quality"
        ],

        # ----------------------------------------------------
        # SCORES
        # ----------------------------------------------------

        "bullish_score": signal_info[
            "bullish_score"
        ],

        "bearish_score": signal_info[
            "bearish_score"
        ],

        "net_score": signal_info[
            "net_score"
        ],

        # ----------------------------------------------------
        # TREND
        # ----------------------------------------------------

        "trend": trend,

        "long_term_trend": long_trend,

        "market_regime": regime,

        "market_structure": structure,

        "candle_pattern": candle,

        # ----------------------------------------------------
        # INDICATORS — FLAT
        # ----------------------------------------------------

        "ma5": indicators["ma5"],

        "ma10": indicators["ma10"],

        "ma20": indicators["ma20"],

        "ema9": indicators["ema9"],

        "ema20": indicators["ema20"],

        "ema50": indicators["ema50"],

        "ema100": indicators["ema100"],

        "ema200": indicators["ema200"],

        "rsi": indicators["rsi"],

        "macd": indicators["macd"],

        "macd_signal": indicators[
            "macd_signal"
        ],

        "macd_histogram": indicators[
            "macd_histogram"
        ],

        "momentum": indicators[
            "momentum"
        ],

        "atr": indicators["atr"],

        "adx": indicators["adx"],

        "plus_di": indicators[
            "plus_di"
        ],

        "minus_di": indicators[
            "minus_di"
        ],

        "vwap": indicators["vwap"],

        # ----------------------------------------------------
        # BOLLINGER
        # ----------------------------------------------------

        "bollinger_middle": indicators[
            "bollinger_middle"
        ],

        "bollinger_upper": indicators[
            "bollinger_upper"
        ],

        "bollinger_lower": indicators[
            "bollinger_lower"
        ],

        "bollinger_width": indicators[
            "bollinger_width"
        ],

        "bollinger_state": bollinger[
            "state"
        ],

        "bollinger_squeeze": bollinger[
            "squeeze"
        ],

        "bollinger_expansion": bollinger[
            "expansion"
        ],

        # ----------------------------------------------------
        # VOLATILITY
        # ----------------------------------------------------

        "volatility_status": volatility[
            "status"
        ],

        "atr_percent": volatility[
            "atr_percent"
        ],

        "volatility_ratio": volatility[
            "ratio"
        ],

        # ----------------------------------------------------
        # VOLUME
        # ----------------------------------------------------

        "volume_status": volume_info[
            "status"
        ],

        "volume_ratio": volume_info[
            "ratio"
        ],

        "volume_available": volume_info[
            "available"
        ],

        # ----------------------------------------------------
        # SUPPORT / RESISTANCE
        # ----------------------------------------------------

        "support": sr[
            "support"
        ],

        "support2": sr[
            "support2"
        ],

        "resistance": sr[
            "resistance"
        ],

        "resistance2": sr[
            "resistance2"
        ],

        "support_distance_percent": safe_float(
            support_distance_percent,
            3
        ),

        "resistance_distance_percent": safe_float(
            resistance_distance_percent,
            3
        ),

        # ----------------------------------------------------
        # PREVIOUS DAY
        # ----------------------------------------------------

        "previous_day_high": previous_day[
            "high"
        ],

        "previous_day_low": previous_day[
            "low"
        ],

        "previous_day_close": previous_day[
            "close"
        ],

        "previous_day_date": previous_day[
            "date"
        ],

        # ----------------------------------------------------
        # PIVOTS
        # ----------------------------------------------------

        "pivot": pivots[
            "pivot"
        ],

        "r1": pivots["r1"],

        "r2": pivots["r2"],

        "s1": pivots["s1"],

        "s2": pivots["s2"],

        # ----------------------------------------------------
        # BREAKOUT
        # ----------------------------------------------------

        "breakout": breakout[
            "status"
        ],

        "breakout_status": breakout[
            "status"
        ],

        "breakout_level": breakout[
            "breakout_level"
        ],

        "breakdown_level": breakout[
            "breakdown_level"
        ],

        "breakout_confirmed": breakout[
            "breakout_confirmed"
        ],

        "breakdown_confirmed": breakout[
            "breakdown_confirmed"
        ],

        "breakout_strength": breakout[
            "strength"
        ],

        "breakout_volume_confirmed": (
            volume_info["available"] and
            volume_info["ratio"] is not None and
            volume_info["ratio"] >= 1.20
        ),

        # ----------------------------------------------------
        # FALSE BREAKOUT
        # ----------------------------------------------------

        "false_breakout": false_breakout[
            "false_breakout"
        ],

        "false_breakdown": false_breakout[
            "false_breakdown"
        ],

        "false_breakout_status": false_breakout[
            "status"
        ],

        # ----------------------------------------------------
        # TRADE PLAN — FLAT
        # ----------------------------------------------------

        "trade_status": trade_plan[
            "status"
        ],

        "entry": trade_plan[
            "entry"
        ],

        "entry_low": trade_plan[
            "entry_low"
        ],

        "entry_high": trade_plan[
            "entry_high"
        ],

        "stop_loss": trade_plan[
            "stop_loss"
        ],

        "target_1": trade_plan[
            "target_1"
        ],

        "target_2": trade_plan[
            "target_2"
        ],

        "trailing_stop": trade_plan[
            "trailing_stop"
        ],

        "risk_points": trade_plan[
            "risk_points"
        ],

        "reward_1_points": trade_plan[
            "reward_1_points"
        ],

        "reward_2_points": trade_plan[
            "reward_2_points"
        ],

        "risk_reward_1": trade_plan[
            "risk_reward_1"
        ],

        "risk_reward_2": trade_plan[
            "risk_reward_2"
        ],

        # ----------------------------------------------------
        # ZONES
        # ----------------------------------------------------

        "entry_zone": zones[
            "entry_zone"
        ],

        "avoid_zone": zones[
            "avoid_zone"
        ],

        # ----------------------------------------------------
        # MTF FLAT
        # ----------------------------------------------------

        "trend_15m": timeframe_data[
            "15m"
        ]["trend"],

        "trend_30m": timeframe_data[
            "30m"
        ]["trend"],

        "trend_1h": timeframe_data[
            "1h"
        ]["trend"],

        "trend_daily": timeframe_data[
            "1D"
        ]["trend"],

        "higher_timeframe_trend": timeframe_data[
            "1h"
        ]["trend"],

        "higher_tf_conflict": htf_conflict,

        "multi_timeframe_alignment": mtf_alignment,

        # ----------------------------------------------------
        # AI EXPLANATION
        # ----------------------------------------------------

        "analysis": analysis_text,

        "analysis_text": analysis_text,

        "bullish_reasons": signal_info[
            "bullish_reasons"
        ],

        "bearish_reasons": signal_info[
            "bearish_reasons"
        ],

        "no_trade_reason": signal_info[
            "no_trade_reason"
        ],

        # ----------------------------------------------------
        # TIMESTAMP
        # ----------------------------------------------------

        "timestamp": now_ist().isoformat(),

        # ----------------------------------------------------
        # NESTED FINAL OBJECTS
        # ----------------------------------------------------

        "indicators": indicators,

        "levels": {
            "support": sr["support"],
            "support2": sr["support2"],
            "resistance": sr["resistance"],
            "resistance2": sr["resistance2"],
            "previous_day_high": previous_day[
                "high"
            ],
            "previous_day_low": previous_day[
                "low"
            ],
            "previous_day_close": previous_day[
                "close"
            ],
            "pivot": pivots["pivot"],
            "r1": pivots["r1"],
            "r2": pivots["r2"],
            "s1": pivots["s1"],
            "s2": pivots["s2"]
        },

        "trade": {
            "status": trade_plan[
                "status"
            ],
            "decision": decision,
            "entry": trade_plan[
                "entry"
            ],
            "entry_low": trade_plan[
                "entry_low"
            ],
            "entry_high": trade_plan[
                "entry_high"
            ],
            "stop_loss": trade_plan[
                "stop_loss"
            ],
            "target_1": trade_plan[
                "target_1"
            ],
            "target_2": trade_plan[
                "target_2"
            ],
            "trailing_stop": trade_plan[
                "trailing_stop"
            ],
            "risk_points": trade_plan[
                "risk_points"
            ],
            "reward_1_points": trade_plan[
                "reward_1_points"
            ],
            "reward_2_points": trade_plan[
                "reward_2_points"
            ],
            "risk_reward_1": trade_plan[
                "risk_reward_1"
            ],
            "risk_reward_2": trade_plan[
                "risk_reward_2"
            ],
            "entry_zone": zones[
                "entry_zone"
            ],
            "avoid_zone": zones[
                "avoid_zone"
            ]
        },

        "timeframes": {
            "15m": timeframe_data[
                "15m"
            ],
            "30m": timeframe_data[
                "30m"
            ],
            "1h": timeframe_data[
                "1h"
            ],
            "1D": timeframe_data[
                "1D"
            ],
            "alignment": mtf_alignment,
            "conflict": htf_conflict
        },

        "breakout_analysis": {
            "status": breakout[
                "status"
            ],
            "breakout_confirmed": breakout[
                "breakout_confirmed"
            ],
            "breakdown_confirmed": breakout[
                "breakdown_confirmed"
            ],
            "breakout_level": breakout[
                "breakout_level"
            ],
            "breakdown_level": breakout[
                "breakdown_level"
            ],
            "strength": breakout[
                "strength"
            ],
            "volume_confirmed": (
                volume_info["available"] and
                volume_info["ratio"] is not None and
                volume_info["ratio"] >= 1.20
            ),
            "false_breakout": false_breakout[
                "false_breakout"
            ],
            "false_breakdown": false_breakout[
                "false_breakdown"
            ]
        },

        "ai": {
            "decision": decision,
            "confidence": confidence,
            "quality": signal_info[
                "quality"
            ],
            "bullish_score": signal_info[
                "bullish_score"
            ],
            "bearish_score": signal_info[
                "bearish_score"
            ],
            "net_score": signal_info[
                "net_score"
            ],
            "market_regime": regime,
            "market_structure": structure,
            "mtf_alignment": mtf_alignment,
            "htf_conflict": htf_conflict,
            "analysis": analysis_text,
            "no_trade_reason": signal_info[
                "no_trade_reason"
            ]
        },

        "system": {
            "status": "ONLINE",
            "data_source": (
                "Yahoo Finance"
            ),
            "symbol": SYMBOL,
            "backend_version": "FINAL-1.0",
            "server_time": now_ist().isoformat()
        }
    }

    return response


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "app": "NIFTY AI TRADER",
        "status": "running",
        "version": "FINAL-1.0",
        "symbol": "NIFTY 50",
        "data_source": "Yahoo Finance (^NSEI)",
        "message": (
            "NIFTY AI Trader Final Backend is running."
        )
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "version": "FINAL-1.0",
        "time": now_ist().isoformat()
    }


# ============================================================
# VERSION
# ============================================================

@app.get("/version")
def version():

    return {
        "app": "NIFTY AI TRADER",
        "version": "FINAL-1.0",
        "backend": "FINAL",
        "symbol": "NIFTY 50",
        "data_source": "Yahoo Finance"
    }


# ============================================================
# MAIN NIFTY ENDPOINT + OI
# ============================================================

@app.get("/nifty")
def nifty():

    result = build_analysis("5m")

    try:

        oi = get_oi_analysis(
            count=40
        )

        result["oi"] = oi

        result["oi_status"] = oi.get(
            "status",
            "ERROR"
        )

        result["oi_source"] = oi.get(
            "source",
            "Kotak Neo"
        )

        result["oi_bias"] = oi.get(
            "oi_bias",
            "UNKNOWN"
        )

        result["oi_score"] = oi.get(
            "oi_score",
            0
        )

        result["pcr"] = oi.get(
            "pcr"
        )

        result["change_oi_pcr"] = oi.get(
            "change_oi_pcr"
        )

        result["max_pain"] = oi.get(
            "max_pain"
        )

        result["oi_support"] = oi.get(
            "oi_support"
        )

        result["oi_resistance"] = oi.get(
            "oi_resistance"
        )

    except Exception as exc:

        result["oi"] = {
            "status": "ERROR",
            "source": "Kotak Neo",
            "message": str(exc)
        }

        result["oi_status"] = "ERROR"
        result["oi_source"] = "Kotak Neo"

        result["oi_bias"] = "UNKNOWN"
        result["oi_score"] = 0

        result["pcr"] = None
        result["change_oi_pcr"] = None
        result["max_pain"] = None

        result["oi_support"] = None
        result["oi_resistance"] = None

    return result


# ============================================================
# NIFTY SUMMARY
# ============================================================

@app.get("/nifty/summary")
def nifty_summary():

    data = build_analysis(
        "5m"
    )

    return {
        "symbol": data.get(
            "symbol"
        ),
        "price": data.get(
            "price"
        ),
        "change": data.get(
            "change"
        ),
        "change_percent": data.get(
            "change_percent"
        ),
        "market_status": data.get(
            "market_status"
        ),
        "trend": data.get(
            "trend"
        ),
        "decision": data.get(
            "decision"
        ),
        "signal": data.get(
            "signal"
        ),
        "confidence": data.get(
            "confidence"
        ),
        "signal_quality": data.get(
            "signal_quality"
        ),
        "bullish_score": data.get(
            "bullish_score"
        ),
        "bearish_score": data.get(
            "bearish_score"
        ),
        "support": data.get(
            "support"
        ),
        "resistance": data.get(
            "resistance"
        ),
        "entry": data.get(
            "entry"
        ),
        "stop_loss": data.get(
            "stop_loss"
        ),
        "target_1": data.get(
            "target_1"
        ),
        "target_2": data.get(
            "target_2"
        ),
        "risk_reward_1": data.get(
            "risk_reward_1"
        ),
        "risk_reward_2": data.get(
            "risk_reward_2"
        ),
        "analysis": data.get(
            "analysis"
        ),
        "timestamp": data.get(
            "timestamp"
        )
    }


# ============================================================
# ALL TIMEFRAMES
# ============================================================

@app.get("/nifty/all-timeframes")
def nifty_all_timeframes():

    timeframe_data = (
        get_higher_timeframe_analysis()
    )

    return {
        "symbol": "NIFTY 50",
        "version": "FINAL-1.0",
        "timeframes": timeframe_data,
        "alignment": calculate_mtf_alignment(
            timeframe_data
        ),
        "conflict": detect_htf_conflict(
            timeframe_data
        ),
        "timestamp": now_ist().isoformat()
    }


# ============================================================
# HISTORY
# ============================================================

@app.get("/nifty/history")
def nifty_history(
    interval: str = "5m",
    limit: int = 250
):

    if interval not in INTERVAL_CONFIG:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported interval: {interval}. "
                f"Supported: {list(INTERVAL_CONFIG.keys())}"
            )
        )

    if limit < 1:
        limit = 1

    if limit > 1000:
        limit = 1000

    df = fetch_history(
        interval
    )

    if df.empty:

        raise HTTPException(
            status_code=503,
            detail="No historical data available."
        )

    df = df.tail(
        limit
    )

    candles = []

    for timestamp, row in df.iterrows():

        candles.append(
            {
                "time": timestamp.isoformat(),
                "timestamp": timestamp.isoformat(),
                "open": safe_float(
                    row["Open"]
                ),
                "high": safe_float(
                    row["High"]
                ),
                "low": safe_float(
                    row["Low"]
                ),
                "close": safe_float(
                    row["Close"]
                ),
                "volume": safe_float(
                    row["Volume"]
                )
            }
        )

    return {
        "symbol": "NIFTY 50",
        "interval": interval,
        "count": len(candles),
        "data": candles,
        "candles": candles,
        "version": "FINAL-1.0",
        "timestamp": now_ist().isoformat()
    }


# ============================================================
# INTERVAL ANALYSIS
# ============================================================

@app.get("/nifty/{interval}")
def nifty_interval(
    interval: str
):

    if interval not in INTERVAL_CONFIG:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported interval: {interval}. "
                f"Supported: {list(INTERVAL_CONFIG.keys())}"
            )
        )

    return build_analysis(
        interval
    )
# ============================================================
# KOTAK OI ENDPOINTS
# ============================================================

# ============================================================
# DHAN API STATUS
# ============================================================

@app.get("/dhan/status")
def dhan_api_status():
    token = os.getenv("DHAN_ACCESS_TOKEN")

    if not token:

        return {
            "status": "ERROR",
            "provider": "Dhan",
            "configured": False,
            "message": "DHAN_ACCESS_TOKEN is not configured.",
            "timestamp": now_ist().isoformat()
        }

    try:

        response = requests.get(
            "https://api.dhan.co/v2/profile",
            headers={
                "access-token": token
            },
            timeout=15
        )

        try:
            data = response.json()
        except Exception:
            data = {}

        if response.status_code != 200:

            return {
                "status": "ERROR",
                "provider": "Dhan",
                "configured": True,
                "http_status": response.status_code,
                "message": data.get(
                    "errorMessage",
                    data.get(
                        "message",
                        "Dhan API request failed."
                    )
                ),
                "timestamp": now_ist().isoformat()
            }

        return {
            "status": "OK",
            "provider": "Dhan",
            "configured": True,
            "dhan_client_id": data.get(
                "dhanClientId"
            ),
            "token_validity": data.get(
                "tokenValidity"
            ),
            "active_segment": data.get(
                "activeSegment"
            ),
            "data_plan": data.get(
                "dataPlan"
            ),
            "data_validity": data.get(
                "dataValidity"
            ),
            "timestamp": now_ist().isoformat()
        }

    except Exception as exc:

        return {
            "status": "ERROR",
            "provider": "Dhan",
            "configured": True,
            "message": str(exc),
            "timestamp": now_ist().isoformat()
        }


# ============================================================
# KOTAK OI ENDPOINTS
# ============================================================

@app.get("/kotak/status")
def kotak_api_status():


    try:

        return {
            "status": "OK",
            **kotak_status(),
            "timestamp": now_ist().isoformat()
        }

    except Exception as exc:

        return {
            "status": "ERROR",
            "provider": "Kotak Neo",
            "configured": False,
            "market_data": False,
            "option_chain": False,
            "order_placement": False,
            "error": str(exc),
            "timestamp": now_ist().isoformat()
        }


# ============================================================
# KOTAK EXPIRY
# ============================================================

@app.get("/kotak/expiry")
def kotak_expiry():

    try:

        expiries = []

        try:

            from kotak_oi import get_nifty_expiries

            expiries = get_nifty_expiries()

        except Exception:

            nearest = get_nearest_expiry()

            if nearest:
                expiries = [nearest]

        return {

            "status": "OK",

            "provider": "Kotak Neo",

            "symbol": "NIFTY",

            "nearest_expiry": (
                expiries[0]
                if expiries
                else None
            ),

            "expiries": expiries,

            "timestamp": now_ist().isoformat()
        }

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=f"Kotak expiry error: {str(exc)}"
        )


# ============================================================
# COMPLETE OI
# ============================================================

@app.get("/oi")
def option_chain_oi(
    expiry: Optional[str] = None,
    count: int = 40
):

    try:

        data = get_oi_analysis(
            expiry=expiry,
            count=count
        )

        return {

            "app": "NIFTY AI TRADER",

            "version": "FINAL-OI-1.0",

            "status": "OK",

            "data_source": "Kotak Neo",

            **data,

            "timestamp": now_ist().isoformat()
        }

    except Exception as exc:

        return {

            "app": "NIFTY AI TRADER",

            "version": "FINAL-OI-1.0",

            "status": "UNAVAILABLE",

            "data_source": "Kotak Neo",

            "error": str(exc),

            "message": (
                "Option Chain/OI data is currently unavailable."
            ),

            "timestamp": now_ist().isoformat()
        }


# ============================================================
# OI SUMMARY
# ============================================================

@app.get("/oi/summary")
def option_chain_oi_summary(
    expiry: Optional[str] = None,
    count: int = 40
):

    try:

        data = get_oi_analysis(
            expiry=expiry,
            count=count
        )

        return {

            "status": "OK",

            "source": "Kotak Neo",

            "symbol": "NIFTY",

            "expiry": data.get(
                "expiry"
            ),

            "total_call_oi": data.get(
                "total_call_oi"
            ),

            "total_put_oi": data.get(
                "total_put_oi"
            ),

            "call_change_oi": data.get(
                "call_change_oi"
            ),

            "put_change_oi": data.get(
                "put_change_oi"
            ),

            "pcr": data.get(
                "pcr"
            ),

            "change_oi_pcr": data.get(
                "change_oi_pcr"
            ),

            "max_pain": data.get(
                "max_pain"
            ),

            "oi_support": data.get(
                "oi_support"
            ),

            "oi_resistance": data.get(
                "oi_resistance"
            ),

            "oi_bias": data.get(
                "oi_bias"
            ),

            "oi_score": data.get(
                "oi_score"
            ),

            "oi_reasons": data.get(
                "oi_reasons"
            ),

            "top_call_oi": data.get(
                "top_call_oi"
            ),

            "top_put_oi": data.get(
                "top_put_oi"
            ),

            "timestamp": now_ist().isoformat()
        }

    except Exception as exc:

        return {

            "status": "UNAVAILABLE",

            "source": "Kotak Neo",

            "symbol": "NIFTY",

            "error": str(exc),

            "timestamp": now_ist().isoformat()
        }


# ============================================================
# OI LEVELS
# ============================================================

@app.get("/oi/levels")
def option_chain_oi_levels(
    expiry: Optional[str] = None,
    count: int = 40
):

    try:

        data = get_oi_analysis(
            expiry=expiry,
            count=count
        )

        return {

            "status": "OK",

            "source": "Kotak Neo",

            "symbol": "NIFTY",

            "expiry": data.get(
                "expiry"
            ),

            "oi_support": data.get(
                "oi_support"
            ),

            "oi_resistance": data.get(
                "oi_resistance"
            ),

            "max_pain": data.get(
                "max_pain"
            ),

            "pcr": data.get(
                "pcr"
            ),

            "change_oi_pcr": data.get(
                "change_oi_pcr"
            ),

            "top_call_oi": data.get(
                "top_call_oi"
            ),

            "top_put_oi": data.get(
                "top_put_oi"
            ),

            "oi_bias": data.get(
                "oi_bias"
            ),

            "oi_score": data.get(
                "oi_score"
            ),

            "timestamp": now_ist().isoformat()
        }

    except Exception as exc:

        return {

            "status": "UNAVAILABLE",

            "source": "Kotak Neo",

            "error": str(exc),

            "timestamp": now_ist().isoformat()
        }


# ============================================================
# STRIKE-WISE OI CHAIN
# ============================================================

@app.get("/oi/chain")
def option_chain_oi_chain(
    expiry: Optional[str] = None,
    count: int = 40
):

    try:

        data = get_oi_analysis(
            expiry=expiry,
            count=count
        )

        return {

            "status": "OK",

            "source": "Kotak Neo",

            "symbol": "NIFTY",

            "expiry": data.get(
                "expiry"
            ),

            "count": len(
                data.get(
                    "chain",
                    []
                )
            ),

            "chain": data.get(
                "chain",
                []
            ),

            "timestamp": now_ist().isoformat()
        }

    except Exception as exc:

        return {

            "status": "UNAVAILABLE",

            "source": "Kotak Neo",

            "symbol": "NIFTY",

            "chain": [],

            "error": str(exc),

            "timestamp": now_ist().isoformat()
        }    


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=10000
    )
