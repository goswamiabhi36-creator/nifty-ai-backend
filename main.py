from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import yfinance as yf
import pandas as pd
import numpy as np

from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo
from typing import Optional


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="NIFTY AI TRADER",
    version="5.1"
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

MAX_LIVE_AGE_SECONDS = 90

INTERVAL_CONFIG = {
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

def now_ist() -> datetime:
    return datetime.now(IST)


def safe_float(value, default=None):
    try:
        if value is None:
            return default

        value = float(value)

        if np.isnan(value) or np.isinf(value):
            return default

        return value

    except Exception:
        return default


def round_value(value, digits=2):
    value = safe_float(value)

    if value is None:
        return None

    return round(value, digits)


# ============================================================
# YFINANCE COLUMN NORMALIZATION
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Handle MultiIndex columns from yfinance
    if isinstance(df.columns, pd.MultiIndex):

        new_columns = []

        for col in df.columns:

            found = None

            for part in col:
                part = str(part).strip()

                if part.lower() in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "adj close",
                    "volume"
                ]:
                    found = part
                    break

            if found is None:
                found = str(col[0])

            new_columns.append(found)

        df.columns = new_columns

    else:
        df.columns = [
            str(c).strip()
            for c in df.columns
        ]

    rename_map = {}

    for col in df.columns:

        lower = col.lower()

        if lower == "open":
            rename_map[col] = "Open"

        elif lower == "high":
            rename_map[col] = "High"

        elif lower == "low":
            rename_map[col] = "Low"

        elif lower == "close":
            rename_map[col] = "Close"

        elif lower in ["adj close", "adj_close"]:
            rename_map[col] = "Adj Close"

        elif lower == "volume":
            rename_map[col] = "Volume"

    df = df.rename(columns=rename_map)

    required = [
        "Open",
        "High",
        "Low",
        "Close"
    ]

    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    if "Volume" not in df.columns:
        df["Volume"] = np.nan

    for col in [
        "Open",
        "High",
        "Low",
        "Close",
        "Volume"
    ]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    df = df.dropna(
        subset=[
            "Open",
            "High",
            "Low",
            "Close"
        ]
    )

    return df


# ============================================================
# FETCH DATA
# ============================================================

def fetch_history(
    interval: str = "5m"
) -> pd.DataFrame:

    if interval not in INTERVAL_CONFIG:
        raise ValueError(
            "Invalid interval. Use 1m, 5m, 15m, 30m, 1h or 1D."
        )

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

        df = normalize_columns(df)

        if df.empty:
            return pd.DataFrame()

        # Make timezone consistent
        try:

            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")

            df.index = df.index.tz_convert(IST)

        except Exception:
            pass

        df = df.sort_index()

        return df

    except Exception:
        return pd.DataFrame()


# ============================================================
# MARKET STATUS
# ============================================================

def get_market_status(
    latest_timestamp
):

    now = now_ist()

    if now.weekday() >= 5:
        return "CLOSED"

    current_time = now.time()

    if current_time < MARKET_OPEN:
        return "CLOSED"

    if current_time > MARKET_CLOSE:
        return "CLOSED"

    try:

        if latest_timestamp is None:
            return "DELAYED"

        if latest_timestamp.tzinfo is None:
            latest_timestamp = latest_timestamp.replace(
                tzinfo=IST
            )

        latest_timestamp = latest_timestamp.astimezone(IST)

        age = (
            now - latest_timestamp
        ).total_seconds()

        if age <= MAX_LIVE_AGE_SECONDS:
            return "LIVE"

        return "DELAYED"

    except Exception:
        return "DELAYED"


# ============================================================
# INDICATORS
# ============================================================

def calculate_indicators(df: pd.DataFrame):

    df = df.copy()

    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    # --------------------------------------------------------
    # Moving averages
    # --------------------------------------------------------

    df["MA5"] = close.rolling(5).mean()
    df["MA10"] = close.rolling(10).mean()
    df["MA20"] = close.rolling(20).mean()

    # --------------------------------------------------------
    # EMA
    # --------------------------------------------------------

    df["EMA9"] = close.ewm(
        span=9,
        adjust=False
    ).mean()

    df["EMA20"] = close.ewm(
        span=20,
        adjust=False
    ).mean()

    df["EMA50"] = close.ewm(
        span=50,
        adjust=False
    ).mean()

    df["EMA100"] = close.ewm(
        span=100,
        adjust=False
    ).mean()

    df["EMA200"] = close.ewm(
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

    df["RSI14"] = 100 - (
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

    df["MACD"] = ema12 - ema26

    df["MACD_SIGNAL"] = df["MACD"].ewm(
        span=9,
        adjust=False
    ).mean()

    df["MACD_HIST"] = (
        df["MACD"] -
        df["MACD_SIGNAL"]
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    df["MOMENTUM"] = close.diff(10)

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

    df["ATR14"] = true_range.rolling(
        14
    ).mean()

    # --------------------------------------------------------
    # ADX + DI
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
        index=df.index
    )

    minus_dm = pd.Series(
        np.where(
            (down_move > up_move) &
            (down_move > 0),
            down_move,
            0
        ),
        index=df.index
    )

    tr14 = true_range.rolling(14).sum()

    plus_di = (
        100 *
        plus_dm.rolling(14).sum() /
        tr14.replace(0, np.nan)
    )

    minus_di = (
        100 *
        minus_dm.rolling(14).sum() /
        tr14.replace(0, np.nan)
    )

    dx = (
        100 *
        (plus_di - minus_di).abs() /
        (plus_di + minus_di).replace(
            0,
            np.nan
        )
    )

    df["PLUS_DI14"] = plus_di
    df["MINUS_DI14"] = minus_di

    df["ADX14"] = dx.rolling(14).mean()

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    volume = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    )

    typical_price = (
        high + low + close
    ) / 3

    volume_valid = (
        volume.notna() &
        (volume > 0)
    )

    if volume_valid.any():

        cumulative_volume = volume.where(
            volume_valid,
            0
        ).cumsum()

        cumulative_pv = (
            typical_price *
            volume.where(
                volume_valid,
                0
            )
        ).cumsum()

        df["VWAP"] = (
            cumulative_pv /
            cumulative_volume.replace(
                0,
                np.nan
            )
        )

    else:

        df["VWAP"] = np.nan

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()

    df["BB_MIDDLE"] = bb_mid
    df["BB_UPPER"] = bb_mid + 2 * bb_std
    df["BB_LOWER"] = bb_mid - 2 * bb_std

    df["BB_WIDTH"] = (
        (
            df["BB_UPPER"] -
            df["BB_LOWER"]
        ) /
        df["BB_MIDDLE"].replace(
            0,
            np.nan
        )
    )

    return df


# ============================================================
# SUPPORT / RESISTANCE
# IMPORTANT:
# CURRENT CANDLE IS EXCLUDED
# ============================================================

def calculate_support_resistance(df):

    if len(df) < 20:
        return {
            "support": None,
            "resistance": None,
            "support2": None,
            "resistance2": None
        }

    # Exclude current candle.
    # This fixes the old breakout calculation problem.
    previous = df.iloc[:-1]

    recent50 = previous.tail(50)
    recent20 = previous.tail(20)

    support = safe_float(
        recent50["Low"].min()
    )

    resistance = safe_float(
        recent50["High"].max()
    )

    support2 = safe_float(
        recent20["Low"].min()
    )

    resistance2 = safe_float(
        recent20["High"].max()
    )

    return {
        "support": support,
        "resistance": resistance,
        "support2": support2,
        "resistance2": resistance2
    }


# ============================================================
# PREVIOUS DAY LEVELS
# ============================================================

def get_previous_day_levels():

    df = fetch_history("1D")

    if df.empty or len(df) < 2:
        return {
            "previous_day_high": None,
            "previous_day_low": None,
            "previous_day_close": None,
            "day_open": None
        }

    today = now_ist().date()

    completed = df.copy()

    try:
        dates = completed.index.date

        # Remove current day if present
        mask = dates < today

        if mask.any():
            completed = completed.loc[mask]

    except Exception:
        pass

    if len(completed) < 1:
        return {
            "previous_day_high": None,
            "previous_day_low": None,
            "previous_day_close": None,
            "day_open": None
        }

    previous = completed.iloc[-1]

    return {
        "previous_day_high": safe_float(
            previous["High"]
        ),
        "previous_day_low": safe_float(
            previous["Low"]
        ),
        "previous_day_close": safe_float(
            previous["Close"]
        ),
        "day_open": None
    }


# ============================================================
# DAY OPEN
# ============================================================

def get_day_open(df):

    if df.empty:
        return None

    try:

        latest_date = df.index[-1].date()

        same_day = df[
            df.index.date == latest_date
        ]

        if not same_day.empty:
            return safe_float(
                same_day.iloc[0]["Open"]
            )

    except Exception:
        pass

    return safe_float(
        df.iloc[-1]["Open"]
    )


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
            "pivot_r1": None,
            "pivot_s1": None,
            "pivot_r2": None,
            "pivot_s2": None
        }

    pivot = (
        previous_high +
        previous_low +
        previous_close
    ) / 3

    r1 = (
        2 * pivot
    ) - previous_low

    s1 = (
        2 * pivot
    ) - previous_high

    r2 = (
        pivot +
        previous_high -
        previous_low
    )

    s2 = (
        pivot -
        previous_high +
        previous_low
    )

    return {
        "pivot": pivot,
        "pivot_r1": r1,
        "pivot_s1": s1,
        "pivot_r2": r2,
        "pivot_s2": s2
    }


# ============================================================
# VOLUME ANALYSIS
# ============================================================

def calculate_volume_status(df):

    if df.empty or "Volume" not in df.columns:
        return {
            "status": "UNAVAILABLE",
            "ratio": None,
            "available": False
        }

    volume = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    )

    valid = volume[
        volume > 0
    ].dropna()

    if len(valid) < 5:
        return {
            "status": "UNAVAILABLE",
            "ratio": None,
            "available": False
        }

    latest = safe_float(
        valid.iloc[-1]
    )

    average = safe_float(
        valid.iloc[:-1].tail(20).mean()
    )

    if latest is None or average is None:
        return {
            "status": "UNAVAILABLE",
            "ratio": None,
            "available": False
        }

    ratio = latest / average if average > 0 else None

    if ratio is None:
        status = "UNAVAILABLE"

    elif ratio >= 1.5:
        status = "HIGH"

    elif ratio <= 0.7:
        status = "LOW"

    else:
        status = "NORMAL"

    return {
        "status": status,
        "ratio": ratio,
        "available": True
    }


# ============================================================
# MARKET STRUCTURE
# ============================================================

def calculate_market_structure(df):

    if len(df) < 10:
        return "UNKNOWN"

    recent = df.tail(10)

    first_half = recent.iloc[:5]
    second_half = recent.iloc[5:]

    first_high = safe_float(
        first_half["High"].max()
    )

    second_high = safe_float(
        second_half["High"].max()
    )

    first_low = safe_float(
        first_half["Low"].min()
    )

    second_low = safe_float(
        second_half["Low"].min()
    )

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

    if (
        second_high > first_high and
        second_low < first_low
    ):
        return "EXPANSION"

    return "RANGE"


# ============================================================
# CANDLE PATTERN
# ============================================================

def detect_candle_pattern(df):

    if len(df) < 3:
        return "NONE"

    current = df.iloc[-1]
    previous = df.iloc[-2]

    o = safe_float(current["Open"])
    h = safe_float(current["High"])
    l = safe_float(current["Low"])
    c = safe_float(current["Close"])

    po = safe_float(previous["Open"])
    pc = safe_float(previous["Close"])

    if None in [o, h, l, c, po, pc]:
        return "NONE"

    body = abs(c - o)
    candle_range = h - l

    if candle_range <= 0:
        return "NONE"

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    body_ratio = body / candle_range

    # Bullish engulfing
    if (
        pc < po and
        c > o and
        o <= pc and
        c >= po
    ):
        return "BULLISH_ENGULFING"

    # Bearish engulfing
    if (
        pc > po and
        c < o and
        o >= pc and
        c <= po
    ):
        return "BEARISH_ENGULFING"

    # Hammer
    if (
        lower_wick >= body * 2 and
        upper_wick <= body * 0.8 and
        body_ratio <= 0.45
    ):
        return "HAMMER"

    # Shooting star
    if (
        upper_wick >= body * 2 and
        lower_wick <= body * 0.8 and
        body_ratio <= 0.45
    ):
        return "SHOOTING_STAR"

    # Strong bullish candle
    if (
        c > o and
        body_ratio >= 0.65
    ):
        return "STRONG_BULLISH"

    # Strong bearish candle
    if (
        c < o and
        body_ratio >= 0.65
    ):
        return "STRONG_BEARISH"

    return "NONE"


# ============================================================
# VOLATILITY ANALYSIS
# ============================================================

def calculate_volatility(df):

    if len(df) < 30:
        return {
            "status": "UNKNOWN",
            "atr_percent": None,
            "atr_ratio": None
        }

    close = safe_float(
        df["Close"].iloc[-1]
    )

    atr = safe_float(
        df["ATR14"].iloc[-1]
    )

    if close is None or atr is None:
        return {
            "status": "UNKNOWN",
            "atr_percent": None,
            "atr_ratio": None
        }

    atr_percent = (
        atr / close
    ) * 100

    atr_series = df["ATR14"].dropna()

    if len(atr_series) < 10:
        return {
            "status": "NORMAL",
            "atr_percent": atr_percent,
            "atr_ratio": None
        }

    median_atr = safe_float(
        atr_series.tail(30).median()
    )

    if median_atr is None or median_atr <= 0:
        return {
            "status": "NORMAL",
            "atr_percent": atr_percent,
            "atr_ratio": None
        }

    ratio = atr / median_atr

    if ratio >= 1.35:
        status = "HIGH"

    elif ratio <= 0.70:
        status = "LOW"

    else:
        status = "NORMAL"

    return {
        "status": status,
        "atr_percent": atr_percent,
        "atr_ratio": ratio
    }


# ============================================================
# BOLLINGER ANALYSIS
# ============================================================

def calculate_bollinger_state(df):

    if len(df) < 30:
        return {
            "squeeze": False,
            "expansion": False
        }

    width = df["BB_WIDTH"].dropna()

    if len(width) < 20:
        return {
            "squeeze": False,
            "expansion": False
        }

    current = safe_float(
        width.iloc[-1]
    )

    median_width = safe_float(
        width.tail(20).median()
    )

    previous = safe_float(
        width.iloc[-2]
    )

    if None in [
        current,
        median_width,
        previous
    ]:
        return {
            "squeeze": False,
            "expansion": False
        }

    squeeze = (
        current <=
        median_width * 0.75
    )

    expansion = (
        current >=
        previous * 1.15
    )

    return {
        "squeeze": bool(squeeze),
        "expansion": bool(expansion)
    }


# ============================================================
# TREND
# ============================================================

def calculate_trend(df):

    if len(df) < 50:
        return "SIDEWAYS"

    latest = df.iloc[-1]

    close = safe_float(
        latest["Close"]
    )

    ema9 = safe_float(
        latest["EMA9"]
    )

    ema20 = safe_float(
        latest["EMA20"]
    )

    ema50 = safe_float(
        latest["EMA50"]
    )

    if None in [
        close,
        ema9,
        ema20,
        ema50
    ]:
        return "SIDEWAYS"

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
# MARKET REGIME
# ============================================================

def calculate_market_regime(df):

    if df.empty:
        return "UNKNOWN"

    latest = df.iloc[-1]

    adx = safe_float(
        latest["ADX14"]
    )

    plus_di = safe_float(
        latest["PLUS_DI14"]
    )

    minus_di = safe_float(
        latest["MINUS_DI14"]
    )

    if None in [
        adx,
        plus_di,
        minus_di
    ]:
        return "UNKNOWN"

    if adx >= 25:

        if plus_di > minus_di:
            return "TREND_UP"

        if minus_di > plus_di:
            return "TREND_DOWN"

    return "RANGE_CHOP"


# ============================================================
# SIGNAL ENGINE V5.1
# ============================================================

def generate_signal(
    df,
    market_status,
    htf_trends,
    sr,
    volume_info,
    market_structure,
    candle_pattern,
    volatility,
    bollinger
):

    if df.empty:
        return {
            "signal": "NEUTRAL",
            "signal_strength": "LOW",
            "confidence": 0,
            "bullish_score": 0,
            "bearish_score": 0,
            "reasons": [],
            "warnings": [
                "No market data available."
            ],
            "signal_quality": "LOW",
            "no_trade_reason": "No data"
        }

    latest = df.iloc[-1]

    close = safe_float(
        latest["Close"]
    )

    ema9 = safe_float(
        latest["EMA9"]
    )

    ema20 = safe_float(
        latest["EMA20"]
    )

    ema50 = safe_float(
        latest["EMA50"]
    )

    rsi = safe_float(
        latest["RSI14"]
    )

    macd = safe_float(
        latest["MACD"]
    )

    macd_signal = safe_float(
        latest["MACD_SIGNAL"]
    )

    momentum = safe_float(
        latest["MOMENTUM"]
    )

    vwap = safe_float(
        latest["VWAP"]
    )

    adx = safe_float(
        latest["ADX14"]
    )

    plus_di = safe_float(
        latest["PLUS_DI14"]
    )

    minus_di = safe_float(
        latest["MINUS_DI14"]
    )

    bullish = 0
    bearish = 0

    reasons = []
    warnings = []

    bullish_confirmations = 0
    bearish_confirmations = 0

    # --------------------------------------------------------
    # EMA 9 / 20
    # --------------------------------------------------------

    if ema9 is not None and ema20 is not None:

        if ema9 > ema20:
            bullish += 1
            bullish_confirmations += 1
            reasons.append(
                "EMA9 above EMA20"
            )

        elif ema9 < ema20:
            bearish += 1
            bearish_confirmations += 1
            reasons.append(
                "EMA9 below EMA20"
            )

    # --------------------------------------------------------
    # EMA 20 / 50
    # --------------------------------------------------------

    if ema20 is not None and ema50 is not None:

        if ema20 > ema50:
            bullish += 1
            bullish_confirmations += 1
            reasons.append(
                "EMA20 above EMA50"
            )

        elif ema20 < ema50:
            bearish += 1
            bearish_confirmations += 1
            reasons.append(
                "EMA20 below EMA50"
            )

    # --------------------------------------------------------
    # RSI
    # --------------------------------------------------------

    if rsi is not None:

        if rsi >= 55:
            bullish += 1
            bullish_confirmations += 1
            reasons.append(
                f"RSI bullish ({rsi:.1f})"
            )

        elif rsi <= 45:
            bearish += 1
            bearish_confirmations += 1
            reasons.append(
                f"RSI bearish ({rsi:.1f})"
            )

    # --------------------------------------------------------
    # MACD
    # --------------------------------------------------------

    if (
        macd is not None and
        macd_signal is not None
    ):

        if macd > macd_signal:
            bullish += 1
            bullish_confirmations += 1
            reasons.append(
                "MACD bullish"
            )

        elif macd < macd_signal:
            bearish += 1
            bearish_confirmations += 1
            reasons.append(
                "MACD bearish"
            )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    if (
        close is not None and
        vwap is not None
    ):

        if close > vwap:
            bullish += 1
            bullish_confirmations += 1
            reasons.append(
                "Price above VWAP"
            )

        elif close < vwap:
            bearish += 1
            bearish_confirmations += 1
            reasons.append(
                "Price below VWAP"
            )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    if momentum is not None:

        if momentum > 0:
            bullish += 1
            bullish_confirmations += 1

        elif momentum < 0:
            bearish += 1
            bearish_confirmations += 1

    # --------------------------------------------------------
    # ADX + DI
    # --------------------------------------------------------

    strong_adx = (
        adx is not None and
        adx >= 25
    )

    if strong_adx:

        if (
            plus_di is not None and
            minus_di is not None
        ):

            if plus_di > minus_di:
                bullish += 2
                bullish_confirmations += 2

                reasons.append(
                    f"+DI dominant with ADX {adx:.1f}"
                )

            elif minus_di > plus_di:
                bearish += 2
                bearish_confirmations += 2

                reasons.append(
                    f"-DI dominant with ADX {adx:.1f}"
                )

    else:

        warnings.append(
            "ADX below 25: trend strength is weak."
        )

    # --------------------------------------------------------
    # MARKET STRUCTURE
    # --------------------------------------------------------

    if market_structure == "HH_HL":

        bullish += 2
        bullish_confirmations += 2

        reasons.append(
            "Higher High / Higher Low structure"
        )

    elif market_structure == "LH_LL":

        bearish += 2
        bearish_confirmations += 2

        reasons.append(
            "Lower High / Lower Low structure"
        )

    elif market_structure == "RANGE":

        warnings.append(
            "Market structure is range-bound."
        )

    # --------------------------------------------------------
    # CANDLE CONFIRMATION
    # --------------------------------------------------------

    bullish_candles = [
        "BULLISH_ENGULFING",
        "HAMMER",
        "STRONG_BULLISH"
    ]

    bearish_candles = [
        "BEARISH_ENGULFING",
        "SHOOTING_STAR",
        "STRONG_BEARISH"
    ]

    if candle_pattern in bullish_candles:

        bullish += 1
        bullish_confirmations += 1

        reasons.append(
            f"Bullish candle: {candle_pattern}"
        )

    elif candle_pattern in bearish_candles:

        bearish += 1
        bearish_confirmations += 1

        reasons.append(
            f"Bearish candle: {candle_pattern}"
        )

    # --------------------------------------------------------
    # BREAKOUT / BREAKDOWN
    # --------------------------------------------------------

    resistance = sr.get(
        "resistance"
    )

    support = sr.get(
        "support"
    )

    breakout = False
    breakdown = False

    if (
        close is not None and
        resistance is not None and
        close > resistance
    ):

        breakout = True

        bullish += 3
        bullish_confirmations += 3

        reasons.append(
            "Resistance breakout"
        )

    if (
        close is not None and
        support is not None and
        close < support
    ):

        breakdown = True

        bearish += 3
        bearish_confirmations += 3

        reasons.append(
            "Support breakdown"
        )

    # --------------------------------------------------------
    # BREAKOUT VOLUME CONFIRMATION
    # --------------------------------------------------------

    breakout_volume_confirmed = False

    if (
        breakout or
        breakdown
    ):

        if volume_info.get("available"):

            ratio = volume_info.get(
                "ratio"
            )

            if ratio is not None and ratio >= 1.2:

                breakout_volume_confirmed = True

                if breakout:
                    bullish += 2
                    bullish_confirmations += 2

                    reasons.append(
                        "Breakout confirmed by volume"
                    )

                elif breakdown:
                    bearish += 2
                    bearish_confirmations += 2

                    reasons.append(
                        "Breakdown confirmed by volume"
                    )

            else:

                warnings.append(
                    "Breakout/breakdown lacks strong volume confirmation."
                )

        else:

            warnings.append(
                "Volume unavailable for NIFTY index."
            )

    # --------------------------------------------------------
    # HIGHER TIMEFRAME ALIGNMENT
    # --------------------------------------------------------

    current_trend = calculate_trend(df)

    trend_values = [
        htf_trends.get("15m"),
        htf_trends.get("30m"),
        htf_trends.get("1h")
    ]

    bullish_htf = sum(
        1 for x in trend_values
        if x == "BULLISH"
    )

    bearish_htf = sum(
        1 for x in trend_values
        if x == "BEARISH"
    )

    if bullish_htf >= 2:

        bullish += 2
        bullish_confirmations += 2

        reasons.append(
            "Higher timeframes support bullish direction"
        )

    elif bearish_htf >= 2:

        bearish += 2
        bearish_confirmations += 2

        reasons.append(
            "Higher timeframes support bearish direction"
        )

    else:

        warnings.append(
            "Higher timeframe confirmation is mixed."
        )

    # --------------------------------------------------------
    # HTF CONFLICT
    # --------------------------------------------------------

    htf_conflict = (
        bullish_htf >= 1 and
        bearish_htf >= 1
    )

    if htf_conflict:

        warnings.append(
            "Higher timeframe trend conflict."
        )

    # --------------------------------------------------------
    # MARKET REGIME
    # --------------------------------------------------------

    regime = calculate_market_regime(df)

    if regime == "RANGE_CHOP":

        warnings.append(
            "Market is in range/choppy regime."
        )

    # --------------------------------------------------------
    # VOLATILITY FILTER
    # --------------------------------------------------------

    if volatility.get("status") == "LOW":

        warnings.append(
            "Volatility is unusually low."
        )

    # --------------------------------------------------------
    # BOLLINGER
    # --------------------------------------------------------

    if bollinger.get("squeeze"):

        warnings.append(
            "Bollinger squeeze detected; breakout may be developing."
        )

    if bollinger.get("expansion"):

        reasons.append(
            "Bollinger volatility expansion"
        )

    # --------------------------------------------------------
    # FINAL SIGNAL
    # --------------------------------------------------------

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
        95,
        max(
            0,
            confidence
        )
    )

    signal = "NEUTRAL"

    # --------------------------------------------------------
    # SAFETY GATES
    # --------------------------------------------------------

    if market_status != "LIVE":

        warnings.append(
            "Market is not LIVE. Trade signal disabled."
        )

    elif htf_conflict:

        warnings.append(
            "Trade blocked because higher timeframes conflict."
        )

    elif regime == "RANGE_CHOP":

        warnings.append(
            "Trade blocked in range/choppy market."
        )

    elif volatility.get("status") == "LOW":

        warnings.append(
            "Trade blocked because volatility is too low."
        )

    else:

        # BUY
        if (
            bullish >= 9 and
            net_score >= 3 and
            bullish_confirmations >= 6
        ):

            signal = "BUY"

        # SELL
        elif (
            bearish >= 9 and
            net_score <= -3 and
            bearish_confirmations >= 6
        ):

            signal = "SELL"

    # --------------------------------------------------------
    # SIGNAL QUALITY
    # --------------------------------------------------------

    if signal in ["BUY", "SELL"]:

        if (
            breakout_volume_confirmed and
            not htf_conflict
        ):
            signal_quality = "HIGH"

        elif (
            abs(net_score) >= 5
        ):
            signal_quality = "MEDIUM"

        else:
            signal_quality = "LOW"

    else:

        signal_quality = "LOW"

    # --------------------------------------------------------
    # NO TRADE REASON
    # --------------------------------------------------------

    if signal == "NEUTRAL":

        if market_status != "LIVE":
            no_trade_reason = "Market not LIVE"

        elif htf_conflict:
            no_trade_reason = "Higher timeframe conflict"

        elif regime == "RANGE_CHOP":
            no_trade_reason = "Range / choppy market"

        elif volatility.get("status") == "LOW":
            no_trade_reason = "Low volatility"

        elif abs(net_score) < 3:
            no_trade_reason = "Insufficient directional confirmation"

        else:
            no_trade_reason = "Signal confirmation incomplete"

    else:

        no_trade_reason = None

    return {
        "signal": signal,
        "signal_strength": (
            "HIGH"
            if signal_quality == "HIGH"
            else (
                "MEDIUM"
                if signal_quality == "MEDIUM"
                else "LOW"
            )
        ),
        "confidence": round(
            confidence,
            1
        ),
        "bullish_score": bullish,
        "bearish_score": bearish,
        "reasons": reasons[-12:],
        "warnings": warnings[-12:],
        "signal_quality": signal_quality,
        "no_trade_reason": no_trade_reason,
        "breakout_volume_confirmed":
            breakout_volume_confirmed,
        "market_regime": regime,
        "htf_conflict": htf_conflict,
        "current_trend": current_trend
    }


# ============================================================
# TRADE PLAN
# EXISTING STYLE PRESERVED
# ============================================================

def calculate_trade_plan(
    df,
    signal
):

    if df.empty:
        return {
            "trade_status": "NO TRADE",
            "entry": None,
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "trailing_stop": None
        }

    if signal not in [
        "BUY",
        "SELL"
    ]:
        return {
            "trade_status": "NO TRADE",
            "entry": None,
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "trailing_stop": None
        }

    latest = df.iloc[-1]

    entry = safe_float(
        latest["Close"]
    )

    atr = safe_float(
        latest["ATR14"]
    )

    if entry is None or atr is None:
        return {
            "trade_status": "NO TRADE",
            "entry": None,
            "stop_loss": None,
            "target_1": None,
            "target_2": None,
            "trailing_stop": None
        }

    if signal == "BUY":

        stop_loss = (
            entry -
            1.2 * atr
        )

        target_1 = (
            entry +
            1.8 * atr
        )

        target_2 = (
            entry +
            2.8 * atr
        )

        trailing_stop = (
            entry -
            1.0 * atr
        )

    else:

        stop_loss = (
            entry +
            1.2 * atr
        )

        target_1 = (
            entry -
            1.8 * atr
        )

        target_2 = (
            entry -
            2.8 * atr
        )

        trailing_stop = (
            entry +
            1.0 * atr
        )

    return {
        "trade_status": "TRADE",
        "entry": round_value(entry),
        "stop_loss": round_value(stop_loss),
        "target_1": round_value(target_1),
        "target_2": round_value(target_2),
        "trailing_stop": round_value(
            trailing_stop
        )
    }


# ============================================================
# HIGHER TIMEFRAME TRENDS
# ============================================================

def get_higher_timeframe_trends():

    result = {
        "15m": "SIDEWAYS",
        "30m": "SIDEWAYS",
        "1h": "SIDEWAYS"
    }

    for interval in [
        "15m",
        "30m",
        "1h"
    ]:

        try:

            df = fetch_history(
                interval
            )

            if not df.empty:

                df = calculate_indicators(
                    df
                )

                result[interval] = calculate_trend(
                    df
                )

        except Exception:
            result[interval] = "SIDEWAYS"

    return result


# ============================================================
# FALSE BREAKOUT DETECTION
# ============================================================

def detect_false_breakout(
    df,
    sr
):

    if len(df) < 3:
        return {
            "false_breakout": False,
            "false_breakdown": False
        }

    previous = df.iloc[-2]
    current = df.iloc[-1]

    previous_close = safe_float(
        previous["Close"]
    )

    current_close = safe_float(
        current["Close"]
    )

    resistance = sr.get(
        "resistance"
    )

    support = sr.get(
        "support"
    )

    false_breakout = False
    false_breakdown = False

    if (
        resistance is not None and
        previous_close is not None and
        current_close is not None
    ):

        current_high = safe_float(
            current["High"]
        )

        if (
            current_high is not None and
            current_high > resistance and
            current_close < resistance
        ):
            false_breakout = True

    if (
        support is not None and
        previous_close is not None and
        current_close is not None
    ):

        current_low = safe_float(
            current["Low"]
        )

        if (
            current_low is not None and
            current_low < support and
            current_close > support
        ):
            false_breakdown = True

    return {
        "false_breakout": false_breakout,
        "false_breakdown": false_breakdown
    }


# ============================================================
# BUILD COMPLETE ANALYSIS
# ============================================================

def build_analysis(
    interval: str = "5m"
):

    df = fetch_history(
        interval
    )

    if df.empty:

        raise HTTPException(
            status_code=503,
            detail="Unable to fetch NIFTY data."
        )

    df = calculate_indicators(
        df
    )

    latest = df.iloc[-1]

    latest_timestamp = df.index[-1]

    market_status = get_market_status(
        latest_timestamp
    )

    price = safe_float(
        latest["Close"]
    )

    previous_close = None

    if len(df) >= 2:

        previous_close = safe_float(
            df["Close"].iloc[-2]
        )

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

    # --------------------------------------------------------
    # Existing indicators
    # --------------------------------------------------------

    ma5 = safe_float(
        latest["MA5"]
    )

    ma10 = safe_float(
        latest["MA10"]
    )

    ma20 = safe_float(
        latest["MA20"]
    )

    ema9 = safe_float(
        latest["EMA9"]
    )

    ema20 = safe_float(
        latest["EMA20"]
    )

    ema50 = safe_float(
        latest["EMA50"]
    )

    ema100 = safe_float(
        latest["EMA100"]
    )

    ema200 = safe_float(
        latest["EMA200"]
    )

    rsi = safe_float(
        latest["RSI14"]
    )

    macd = safe_float(
        latest["MACD"]
    )

    macd_signal = safe_float(
        latest["MACD_SIGNAL"]
    )

    macd_hist = safe_float(
        latest["MACD_HIST"]
    )

    momentum = safe_float(
        latest["MOMENTUM"]
    )

    atr = safe_float(
        latest["ATR14"]
    )

    adx = safe_float(
        latest["ADX14"]
    )

    plus_di = safe_float(
        latest["PLUS_DI14"]
    )

    minus_di = safe_float(
        latest["MINUS_DI14"]
    )

    vwap = safe_float(
        latest["VWAP"]
    )

    bb_middle = safe_float(
        latest["BB_MIDDLE"]
    )

    bb_upper = safe_float(
        latest["BB_UPPER"]
    )

    bb_lower = safe_float(
        latest["BB_LOWER"]
    )

    # --------------------------------------------------------
    # Support resistance
    # --------------------------------------------------------

    sr = calculate_support_resistance(
        df
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    volume_info = calculate_volume_status(
        df
    )

    # --------------------------------------------------------
    # New analysis
    # --------------------------------------------------------

    market_structure = (
        calculate_market_structure(df)
    )

    candle_pattern = (
        detect_candle_pattern(df)
    )

    volatility = (
        calculate_volatility(df)
    )

    bollinger = (
        calculate_bollinger_state(df)
    )

    previous_day = (
        get_previous_day_levels()
    )

    day_open = get_day_open(
        df
    )

    if day_open is None:
        day_open = previous_day.get(
            "day_open"
        )

    previous_day_high = (
        previous_day.get(
            "previous_day_high"
        )
    )

    previous_day_low = (
        previous_day.get(
            "previous_day_low"
        )
    )

    previous_day_close = (
        previous_day.get(
            "previous_day_close"
        )
    )

    pivots = calculate_pivots(
        previous_day_high,
        previous_day_low,
        previous_day_close
    )

    # --------------------------------------------------------
    # Higher timeframe
    # --------------------------------------------------------

    htf_trends = (
        get_higher_timeframe_trends()
    )

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    signal_data = generate_signal(
        df=df,
        market_status=market_status,
        htf_trends=htf_trends,
        sr=sr,
        volume_info=volume_info,
        market_structure=market_structure,
        candle_pattern=candle_pattern,
        volatility=volatility,
        bollinger=bollinger
    )

    signal = signal_data["signal"]

    # --------------------------------------------------------
    # False breakout
    # --------------------------------------------------------

    false_breakout_data = (
        detect_false_breakout(
            df,
            sr
        )
    )

    if false_breakout_data[
        "false_breakout"
    ]:

        signal_data["warnings"].append(
            "Possible false breakout detected."
        )

    if false_breakout_data[
        "false_breakdown"
    ]:

        signal_data["warnings"].append(
            "Possible false breakdown detected."
        )

    # --------------------------------------------------------
    # Trade plan
    # --------------------------------------------------------

    trade_plan = calculate_trade_plan(
        df,
        signal
        if market_status == "LIVE"
        else "NEUTRAL"
    )

    # --------------------------------------------------------
    # Data age
    # --------------------------------------------------------

    try:

        latest_aware = latest_timestamp

        if latest_aware.tzinfo is None:
            latest_aware = latest_aware.replace(
                tzinfo=IST
            )

        age = (
            now_ist() -
            latest_aware.astimezone(IST)
        ).total_seconds()

    except Exception:
        age = None

    # --------------------------------------------------------
    # Trend
    # --------------------------------------------------------

    trend = calculate_trend(
        df
    )

    # Preserve old higher timeframe field
    higher_timeframe_trend = (
        htf_trends.get(
            "15m",
            "SIDEWAYS"
        )
    )

    higher_tf_conflict = (
        signal_data.get(
            "htf_conflict",
            False
        )
    )

    # --------------------------------------------------------
    # Volume status
    # --------------------------------------------------------

    volume_status = volume_info.get(
        "status",
        "UNAVAILABLE"
    )

    # --------------------------------------------------------
    # Signal strength
    # --------------------------------------------------------

    signal_strength = signal_data.get(
        "signal_strength",
        "LOW"
    )

    # --------------------------------------------------------
    # Main response
    # --------------------------------------------------------

    response = {

        # ====================================================
        # EXISTING FIELDS
        # ====================================================

        "symbol": "NIFTY 50",

        "price": round_value(
            price
        ),

        "previous_close": round_value(
            previous_close
        ),

        "change": round_value(
            change
        ),

        "change_percent": round_value(
            change_percent
        ),

        "market_status": market_status,

        "data_age_seconds": round_value(
            age,
            1
        ),

        "latest_data_time": (
            latest_timestamp.isoformat()
            if latest_timestamp is not None
            else None
        ),

        "trend": trend,

        "higher_timeframe_trend":
            higher_timeframe_trend,

        "higher_tf_conflict":
            higher_tf_conflict,

        "signal": signal,

        "signal_strength":
            signal_strength,

        "confidence":
            signal_data.get(
                "confidence",
                0
            ),

        "bullish_score":
            signal_data.get(
                "bullish_score",
                0
            ),

        "bearish_score":
            signal_data.get(
                "bearish_score",
                0
            ),

        "volume_status":
            volume_status,

        "trade_status":
            trade_plan.get(
                "trade_status",
                "NO TRADE"
            ),

        "analysis_version":
            "V5.1",

        # ====================================================
        # OLD INDICATORS
        # ====================================================

        "ma5": round_value(ma5),
        "ma10": round_value(ma10),
        "ma20": round_value(ma20),

        "ema9": round_value(ema9),
        "ema20": round_value(ema20),
        "ema50": round_value(ema50),
        "ema100": round_value(ema100),
        "ema200": round_value(ema200),

        "rsi14": round_value(rsi),
        "macd": round_value(macd),
        "macd_signal": round_value(macd_signal),
        "macd_histogram": round_value(macd_hist),

        "momentum": round_value(momentum),
        "atr14": round_value(atr),
        "adx14": round_value(adx),

        "vwap": round_value(vwap),

        "bollinger_middle":
            round_value(bb_middle),

        "bollinger_upper":
            round_value(bb_upper),

        "bollinger_lower":
            round_value(bb_lower),

        # ====================================================
        # SUPPORT / RESISTANCE
        # ====================================================

        "support":
            round_value(
                sr.get("support")
            ),

        "resistance":
            round_value(
                sr.get("resistance")
            ),

        "support2":
            round_value(
                sr.get("support2")
            ),

        "resistance2":
            round_value(
                sr.get("resistance2")
            ),

        # ====================================================
        # TRADE PLAN
        # ====================================================

        "entry":
            trade_plan.get("entry"),

        "stop_loss":
            trade_plan.get("stop_loss"),

        "target_1":
            trade_plan.get("target_1"),

        "target_2":
            trade_plan.get("target_2"),

        "trailing_stop":
            trade_plan.get("trailing_stop"),

        # ====================================================
        # REASONS / WARNINGS
        # ====================================================

        "reasons":
            signal_data.get(
                "reasons",
                []
            ),

        "warnings":
            signal_data.get(
                "warnings",
                []
            ),

        # ====================================================
        # NEW V5.1 ACCURACY DATA
        # ====================================================

        "plus_di_14":
            round_value(plus_di),

        "minus_di_14":
            round_value(minus_di),

        "market_regime":
            signal_data.get(
                "market_regime",
                "UNKNOWN"
            ),

        "market_structure":
            market_structure,

        "candle_pattern":
            candle_pattern,

        "breakout_volume_confirmed":
            signal_data.get(
                "breakout_volume_confirmed",
                False
            ),

        "bollinger_squeeze":
            bollinger.get(
                "squeeze",
                False
            ),

        "bollinger_expansion":
            bollinger.get(
                "expansion",
                False
            ),

        "volatility_status":
            volatility.get(
                "status",
                "UNKNOWN"
            ),

        "atr_percent":
            round_value(
                volatility.get(
                    "atr_percent"
                ),
                3
            ),

        "atr_ratio":
            round_value(
                volatility.get(
                    "atr_ratio"
                ),
                2
            ),

        # ====================================================
        # MULTI TIMEFRAME
        # ====================================================

        "trend_15m":
            htf_trends.get(
                "15m",
                "SIDEWAYS"
            ),

        "trend_30m":
            htf_trends.get(
                "30m",
                "SIDEWAYS"
            ),

        "trend_1h":
            htf_trends.get(
                "1h",
                "SIDEWAYS"
            ),

        "multi_timeframe_alignment": (
            "BULLISH"
            if (
                sum(
                    1
                    for x in htf_trends.values()
                    if x == "BULLISH"
                ) >= 2
            )
            else (
                "BEARISH"
                if (
                    sum(
                        1
                        for x in htf_trends.values()
                        if x == "BEARISH"
                    ) >= 2
                )
                else "MIXED"
            )
        ),

        # ====================================================
        # PREVIOUS DAY LEVELS
        # ====================================================

        "previous_day_high":
            round_value(
                previous_day_high
            ),

        "previous_day_low":
            round_value(
                previous_day_low
            ),

        "previous_day_close":
            round_value(
                previous_day_close
            ),

        "day_open":
            round_value(
                day_open
            ),

        # ====================================================
        # PIVOTS
        # ====================================================

        "pivot":
            round_value(
                pivots.get("pivot")
            ),

        "pivot_r1":
            round_value(
                pivots.get("pivot_r1")
            ),

        "pivot_s1":
            round_value(
                pivots.get("pivot_s1")
            ),

        "pivot_r2":
            round_value(
                pivots.get("pivot_r2")
            ),

        "pivot_s2":
            round_value(
                pivots.get("pivot_s2")
            ),

        # ====================================================
        # EXTRA STATUS
        # ====================================================

        "volume_ratio":
            round_value(
                volume_info.get(
                    "ratio"
                ),
                2
            ),

        "volume_available":
            volume_info.get(
                "available",
                False
            ),

        "false_breakout":
            false_breakout_data.get(
                "false_breakout",
                False
            ),

        "false_breakdown":
            false_breakout_data.get(
                "false_breakdown",
                False
            ),

        "signal_quality":
            signal_data.get(
                "signal_quality",
                "LOW"
            ),

        "no_trade_reason":
            signal_data.get(
                "no_trade_reason"
            )
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
        "version": "5.1",
        "symbol": "NIFTY 50",
        "message": "NIFTY AI Trader backend is running."
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "version": "5.1",
        "time": datetime.utcnow().isoformat()
    }


# ============================================================
# MAIN NIFTY ENDPOINT
# ============================================================

@app.get("/nifty")
def nifty():

    return build_analysis(
        "5m"
    )


# ============================================================
# HISTORY ENDPOINT
# IMPORTANT: MUST BE BEFORE /nifty/{interval}
# ============================================================

@app.get("/nifty/history")
def nifty_history(
    interval: str = "5m"
):

    if interval not in INTERVAL_CONFIG:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid interval. "
                "Use 1m, 5m, 15m, 30m, 1h or 1D."
            )
        )

    df = fetch_history(
        interval
    )

    if df.empty:

        raise HTTPException(
            status_code=503,
            detail="Unable to fetch NIFTY history."
        )

    candles = []

    for timestamp, row in df.iterrows():

        candles.append({
            "timestamp":
                timestamp.isoformat(),

            "open":
                round_value(
                    row["Open"]
                ),

            "high":
                round_value(
                    row["High"]
                ),

            "low":
                round_value(
                    row["Low"]
                ),

            "close":
                round_value(
                    row["Close"]
                ),

            "volume":
                round_value(
                    row["Volume"]
                )
        })

    return {
        "symbol": "NIFTY 50",
        "interval": interval,
        "count": len(candles),
        "candles": candles,
        "analysis_version": "V5.1"
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
                "Invalid interval. "
                "Use 1m, 5m, 15m, 30m, 1h or 1D."
            )
        )

    return build_analysis(
        interval
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=10000
        )
