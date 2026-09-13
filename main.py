from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import math
from datetime import datetime, timezone, time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Nifty AI Trader Backend",
    version="5.0"
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

# Opening range:
# First 30 minutes = 09:15 to 09:45
OPENING_RANGE_MINUTES = 30

INTERVAL_PERIODS = {
    "1m": "1d",
    "5m": "5d",
    "15m": "1mo",
    "30m": "1mo",
    "1h": "3mo",
    "1D": "1y",
}

ALLOWED_INTERVALS = [
    "1m",
    "5m",
    "15m",
    "30m",
    "1h",
    "1D"
]

MULTI_TIMEFRAMES = [
    "5m",
    "15m",
    "30m",
    "1h"
]


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


def clamp(value, low, high):
    return max(low, min(high, value))


# ============================================================
# NORMALIZE YFINANCE DATA
# ============================================================

def normalize_columns(df):

    if df is None or df.empty:
        return df

    result = df.copy()

    if isinstance(result.columns, pd.MultiIndex):

        new_columns = []

        for col in result.columns:

            if isinstance(col, tuple):

                selected = None

                for item in col:

                    text = str(item).strip().lower()

                    if text in [
                        "open",
                        "high",
                        "low",
                        "close",
                        "adj close",
                        "volume"
                    ]:
                        selected = text
                        break

                if selected is None:
                    selected = str(col[0])

                new_columns.append(selected)

            else:
                new_columns.append(str(col))

        result.columns = new_columns

    result.columns = [
        str(c).strip().lower()
        for c in result.columns
    ]

    result = result.rename(
        columns={
            "adj close": "adj_close"
        }
    )

    required = [
        "open",
        "high",
        "low",
        "close"
    ]

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

        raise ValueError(
            "Invalid interval. "
            "Use 1m, 5m, 15m, 30m, 1h or 1D."
        )

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

    for column in [
        "open",
        "high",
        "low",
        "close",
        "volume"
    ]:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    df = df.dropna(
        subset=[
            "open",
            "high",
            "low",
            "close"
        ]
    )

    if df.empty:

        raise RuntimeError(
            "Market data contains no valid candles."
        )

    df = df.sort_index()

    return df


# ============================================================
# TIMESTAMP HELPERS
# ============================================================

def get_latest_timestamp(df):

    if df is None or df.empty:
        return None

    ts = df.index[-1]

    try:

        if isinstance(ts, pd.Timestamp):

            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")

            return ts.to_pydatetime().astimezone(IST)

        if isinstance(ts, datetime):

            if ts.tzinfo is None:
                ts = ts.replace(
                    tzinfo=timezone.utc
                )

            return ts.astimezone(IST)

        return None

    except Exception:

        return None


def get_data_age_seconds(df):

    latest = get_latest_timestamp(df)

    if latest is None:
        return None

    current = now_ist()

    age = (
        current - latest
    ).total_seconds()

    return max(
        0.0,
        age
    )


# ============================================================
# MARKET STATUS
# ============================================================

def calculate_market_status(df):

    current = now_ist()

    market_session = is_market_hours(
        current
    )

    latest_timestamp = get_latest_timestamp(
        df
    )

    age_seconds = get_data_age_seconds(
        df
    )

    if latest_timestamp is None:

        return {
            "market_status": "UNKNOWN",
            "data_age_seconds": None,
            "latest_data_time": None
        }

    if not market_session:

        return {
            "market_status": "CLOSED",
            "data_age_seconds":
                round(age_seconds, 1)
                if age_seconds is not None
                else None,

            "latest_data_time":
                latest_timestamp.isoformat()
        }

    if age_seconds is None:

        return {
            "market_status": "UNKNOWN",
            "data_age_seconds": None,
            "latest_data_time":
                latest_timestamp.isoformat()
        }

    if age_seconds <= MAX_LIVE_AGE_SECONDS:

        status = "LIVE"

    else:

        status = "DELAYED"

    return {
        "market_status": status,

        "data_age_seconds":
            round(
                age_seconds,
                1
            ),

        "latest_data_time":
            latest_timestamp.isoformat()
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

    result["ma_5"] = close.rolling(
        5
    ).mean()

    result["ma_10"] = close.rolling(
        10
    ).mean()

    result["ma_20"] = close.rolling(
        20
    ).mean()

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

    gain = delta.clip(
        lower=0
    )

    loss = -delta.clip(
        upper=0
    )

    avg_gain = gain.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / 14,
        adjust=False
    ).mean()

    rs = (
        avg_gain /
        avg_loss.replace(
            0,
            np.nan
        )
    )

    result["rsi_14"] = (
        100 -
        (
            100 /
            (1 + rs)
        )
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

    result["macd"] = (
        ema12 - ema26
    )

    result["macd_signal"] = (
        result["macd"].ewm(
            span=9,
            adjust=False
        ).mean()
    )

    result["macd_histogram"] = (
        result["macd"] -
        result["macd_signal"]
    )

    # --------------------------------------------------------
    # Momentum
    # --------------------------------------------------------

    result["momentum"] = close.diff(
        10
    )

    # --------------------------------------------------------
    # ATR
    # --------------------------------------------------------

    previous_close = close.shift(
        1
    )

    tr1 = high - low

    tr2 = (
        high -
        previous_close
    ).abs()

    tr3 = (
        low -
        previous_close
    ).abs()

    true_range = pd.concat(
        [
            tr1,
            tr2,
            tr3
        ],
        axis=1
    ).max(
        axis=1
    )

    result["atr_14"] = (
        true_range.rolling(
            14
        ).mean()
    )

    # --------------------------------------------------------
    # ADX
    # --------------------------------------------------------

    up_move = high.diff()

    down_move = -low.diff()

    plus_dm = np.where(
        (
            up_move >
            down_move
        ) &
        (
            up_move > 0
        ),
        up_move,
        0
    )

    minus_dm = np.where(
        (
            down_move >
            up_move
        ) &
        (
            down_move > 0
        ),
        down_move,
        0
    )

    atr14 = result[
        "atr_14"
    ]

    plus_di = (
        100 *
        pd.Series(
            plus_dm,
            index=result.index
        ).rolling(
            14
        ).mean() /
        atr14
    )

    minus_di = (
        100 *
        pd.Series(
            minus_dm,
            index=result.index
        ).rolling(
            14
        ).mean() /
        atr14
    )

    dx = (
        100 *
        (
            plus_di -
            minus_di
        ).abs() /
        (
            plus_di +
            minus_di
        ).replace(
            0,
            np.nan
        )
    )

    result["adx_14"] = (
        dx.rolling(
            14
        ).mean()
    )

    # --------------------------------------------------------
    # VWAP
    # --------------------------------------------------------

    typical_price = (
        high +
        low +
        close
    ) / 3

    volume = (
        result["volume"]
        .fillna(0)
    )

    cumulative_volume = (
        volume.cumsum()
    )

    cumulative_value = (
        typical_price *
        volume
    ).cumsum()

    result["vwap"] = np.where(
        cumulative_volume > 0,
        cumulative_value /
        cumulative_volume,
        typical_price
    )

    result["vwap"] = pd.Series(
        result["vwap"],
        index=result.index
    )

    # --------------------------------------------------------
    # Bollinger Bands
    # --------------------------------------------------------

    middle = close.rolling(
        20
    ).mean()

    std = close.rolling(
        20
    ).std()

    result[
        "bollinger_middle"
    ] = middle

    result[
        "bollinger_upper"
    ] = (
        middle +
        2 * std
    )

    result[
        "bollinger_lower"
    ] = (
        middle -
        2 * std
    )

    # --------------------------------------------------------
    # Bollinger width
    # --------------------------------------------------------

    result["bollinger_width"] = np.where(
        middle != 0,
        (
            result["bollinger_upper"] -
            result["bollinger_lower"]
        ) / middle * 100,
        np.nan
    )

    return result


# ============================================================
# TREND
# ============================================================

def calculate_trend(row):

    close = safe_float(
        row["close"]
    )

    ema9 = safe_float(
        row["ema_9"]
    )

    ema20 = safe_float(
        row["ema_20"]
    )

    ema50 = safe_float(
        row["ema_50"]
    )

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
# SUPPORT / RESISTANCE
# ============================================================

def calculate_support_resistance(df):

    if len(df) < 12:

        return (
            None,
            None,
            None,
            None
        )

    # IMPORTANT:
    # Current candle is excluded.
    # This makes breakout detection meaningful.

    previous = df.iloc[:-1]

    recent_50 = previous.tail(50)

    if recent_50.empty:

        return (
            None,
            None,
            None,
            None
        )

    support = safe_float(
        recent_50[
            "low"
        ].tail(10).min()
    )

    resistance = safe_float(
        recent_50[
            "high"
        ].tail(10).max()
    )

    support2 = safe_float(
        recent_50[
            "low"
        ].tail(20).min()
    )

    resistance2 = safe_float(
        recent_50[
            "high"
        ].tail(20).max()
    )

    return (
        support,
        support2,
        resistance,
        resistance2
    )


# ============================================================
# VOLUME ANALYSIS
# ============================================================

def calculate_volume_status(df):

    volume = (
        df["volume"]
        .dropna()
    )

    if volume.empty:
        return "UNAVAILABLE", None

    if len(volume) < 20:
        return "UNAVAILABLE", None

    latest_volume = safe_float(
        volume.iloc[-1]
    )

    average_volume = safe_float(
        volume.iloc[-21:-1].mean()
    )

    if (
        latest_volume is None or
        average_volume is None or
        average_volume <= 0
    ):

        return "UNAVAILABLE", None

    ratio = (
        latest_volume /
        average_volume
    )

    if ratio >= 1.5:
        status = "HIGH"

    elif ratio <= 0.7:
        status = "LOW"

    else:
        status = "NORMAL"

    return (
        status,
        round(
            ratio,
            2
        )
    )


def calculate_volume_confirmation(
    df,
    direction
):

    if len(df) < 21:

        return {
            "confirmed": False,
            "status": "UNAVAILABLE",
            "ratio": None,
            "reason": "Insufficient volume data"
        }

    latest = df.iloc[-1]

    volume = safe_float(
        latest["volume"]
    )

    historical = (
        df["volume"]
        .iloc[-21:-1]
        .dropna()
    )

    if (
        volume is None or
        historical.empty
    ):

        return {
            "confirmed": False,
            "status": "UNAVAILABLE",
            "ratio": None,
            "reason": "Volume unavailable"
        }

    average = safe_float(
        historical.mean()
    )

    if average is None or average <= 0:

        return {
            "confirmed": False,
            "status": "UNAVAILABLE",
            "ratio": None,
            "reason": "Volume average unavailable"
        }

    ratio = volume / average

    close = safe_float(
        latest["close"]
    )

    open_price = safe_float(
        latest["open"]
    )

    if (
        close is None or
        open_price is None
    ):

        return {
            "confirmed": False,
            "status": "UNKNOWN",
            "ratio": round(ratio, 2),
            "reason": "Invalid candle"
        }

    if direction == "BULLISH":

        candle_direction = close > open_price

    else:

        candle_direction = close < open_price

    confirmed = (
        ratio >= 1.15 and
        candle_direction
    )

    if ratio >= 1.5:

        status = "HIGH"

    elif ratio >= 1.15:

        status = "CONFIRMING"

    elif ratio < 0.7:

        status = "LOW"

    else:

        status = "NORMAL"

    if confirmed:

        reason = "Volume confirms direction"

    elif ratio < 0.7:

        reason = "Low volume - weak confirmation"

    else:

        reason = "Volume confirmation weak"

    return {
        "confirmed": confirmed,
        "status": status,
        "ratio": round(ratio, 2),
        "reason": reason
    }


# ============================================================
# PREVIOUS DAY LEVELS
# ============================================================

def calculate_previous_day_levels():

    try:

        daily = fetch_history(
            "1D"
        )

        if daily is None or daily.empty:

            return {
                "previous_day_high": None,
                "previous_day_low": None,
                "previous_day_close": None
            }

        daily = daily.copy()

        if len(daily) < 2:

            return {
                "previous_day_high": None,
                "previous_day_low": None,
                "previous_day_close": None
            }

        previous_day = daily.iloc[-2]

        return {
            "previous_day_high":
                safe_float(
                    previous_day["high"]
                ),

            "previous_day_low":
                safe_float(
                    previous_day["low"]
                ),

            "previous_day_close":
                safe_float(
                    previous_day["close"]
                )
        }

    except Exception:

        return {
            "previous_day_high": None,
            "previous_day_low": None,
            "previous_day_close": None
        }


# ============================================================
# WEEKLY LEVELS
# ============================================================

def calculate_weekly_levels():

    try:

        daily = fetch_history(
            "1D"
        )

        if daily is None or daily.empty:

            return {
                "weekly_high": None,
                "weekly_low": None
            }

        daily = daily.copy()

        daily.index = pd.to_datetime(
            daily.index
        )

        if daily.index.tz is None:

            daily.index = daily.index.tz_localize(
                "UTC"
            )

        daily.index = daily.index.tz_convert(
            IST
        )

        current_date = now_ist().date()

        week_start = (
            current_date -
            pd.Timedelta(
                days=current_date.weekday()
            )
        )

        current_week = daily[
            daily.index.date >= week_start
        ]

        # If current week data is unavailable,
        # use latest available week.

        if current_week.empty:

            latest_date = daily.index[-1].date()

            latest_week_start = (
                latest_date -
                pd.Timedelta(
                    days=latest_date.weekday()
                )
            )

            current_week = daily[
                daily.index.date >=
                latest_week_start
            ]

        if current_week.empty:

            return {
                "weekly_high": None,
                "weekly_low": None
            }

        return {
            "weekly_high":
                safe_float(
                    current_week["high"].max()
                ),

            "weekly_low":
                safe_float(
                    current_week["low"].min()
                )
        }

    except Exception:

        return {
            "weekly_high": None,
            "weekly_low": None
        }


# ============================================================
# OPENING RANGE
# ============================================================

def calculate_opening_range(df):

    result = {
        "opening_range_high": None,
        "opening_range_low": None,
        "opening_range_breakout": False,
        "opening_range_breakdown": False
    }

    try:

        if df is None or df.empty:
            return result

        working = df.copy()

        working.index = pd.to_datetime(
            working.index
        )

        if working.index.tz is None:

            working.index = (
                working.index.tz_localize(
                    "UTC"
                )
            )

        working.index = working.index.tz_convert(
            IST
        )

        # Determine latest trading session in data.
        session_date = working.index[-1].date()

        session = working[
            working.index.date ==
            session_date
        ]

        if session.empty:
            return result

        start_minutes = (
            9 * 60 + 15
        )

        end_minutes = (
            start_minutes +
            OPENING_RANGE_MINUTES
        )

        session_minutes = (
            session.index.hour * 60 +
            session.index.minute
        )

        opening = session[
            (
                session_minutes >=
                start_minutes
            ) &
            (
                session_minutes <=
                end_minutes
            )
        ]

        if opening.empty:
            return result

        opening_high = safe_float(
            opening["high"].max()
        )

        opening_low = safe_float(
            opening["low"].min()
        )

        result[
            "opening_range_high"
        ] = opening_high

        result[
            "opening_range_low"
        ] = opening_low

        latest_close = safe_float(
            session["close"].iloc[-1]
        )

        if (
            latest_close is not None and
            opening_high is not None and
            latest_close > opening_high
        ):

            result[
                "opening_range_breakout"
            ] = True

        if (
            latest_close is not None and
            opening_low is not None and
            latest_close < opening_low
        ):

            result[
                "opening_range_breakdown"
            ] = True

        return result

    except Exception:

        return result


# ============================================================
# MARKET REGIME
# ============================================================

def calculate_market_regime(
    row,
    df
):

    close = safe_float(
        row["close"]
    )

    ema20 = safe_float(
        row["ema_20"]
    )

    ema50 = safe_float(
        row["ema_50"]
    )

    adx = safe_float(
        row["adx_14"]
    )

    atr = safe_float(
        row["atr_14"]
    )

    bollinger_width = safe_float(
        row["bollinger_width"]
    )

    if close is None:

        return {
            "market_regime": "UNKNOWN",
            "regime_strength": "LOW",
            "regime_score": 0
        }

    if (
        adx is not None and
        adx >= 25
    ):

        if (
            ema20 is not None and
            ema50 is not None and
            close > ema20 and
            ema20 > ema50
        ):

            return {
                "market_regime":
                    "TRENDING BULLISH",
                "regime_strength":
                    "HIGH" if adx >= 30
                    else "MEDIUM",
                "regime_score":
                    int(clamp(adx, 25, 50))
            }

        if (
            ema20 is not None and
            ema50 is not None and
            close < ema20 and
            ema20 < ema50
        ):

            return {
                "market_regime":
                    "TRENDING BEARISH",
                "regime_strength":
                    "HIGH" if adx >= 30
                    else "MEDIUM",
                "regime_score":
                    int(clamp(adx, 25, 50))
            }

    if (
        bollinger_width is not None and
        bollinger_width < 1.0
    ):

        return {
            "market_regime":
                "LOW VOLATILITY",
            "regime_strength":
                "MEDIUM",
            "regime_score":
                20
        }

    if (
        bollinger_width is not None and
        bollinger_width > 3.5
    ):

        return {
            "market_regime":
                "HIGH VOLATILITY",
            "regime_strength":
                "HIGH",
            "regime_score":
                40
        }

    return {
        "market_regime":
            "SIDEWAYS",
        "regime_strength":
            "LOW",
        "regime_score":
            15
    }


# ============================================================
# MULTI-TIMEFRAME ANALYSIS
# ============================================================

def calculate_multi_timeframe():

    result = {}

    bullish_count = 0
    bearish_count = 0
    sideways_count = 0

    for interval in MULTI_TIMEFRAMES:

        try:

            df = fetch_history(
                interval
            )

            df = calculate_indicators(
                df
            )

            row = df.iloc[-1]

            trend = calculate_trend(
                row
            )

            result[
                interval
            ] = trend

            if trend == "BULLISH":
                bullish_count += 1

            elif trend == "BEARISH":
                bearish_count += 1

            else:
                sideways_count += 1

        except Exception:

            result[
                interval
            ] = "UNKNOWN"

    known = (
        bullish_count +
        bearish_count +
        sideways_count
    )

    if known == 0:

        overall = "UNKNOWN"

    elif bullish_count >= 3:

        overall = "BULLISH"

    elif bearish_count >= 3:

        overall = "BEARISH"

    elif (
        bullish_count > bearish_count
    ):

        overall = "BULLISH"

    elif (
        bearish_count > bullish_count
    ):

        overall = "BEARISH"

    else:

        overall = "MIXED"

    return {
        "timeframes": result,
        "overall": overall,
        "bullish_count": bullish_count,
        "bearish_count": bearish_count,
        "sideways_count": sideways_count
    }


# ============================================================
# MULTI-TIMEFRAME SCORE
# ============================================================

def calculate_mtf_score(
    mtf_data,
    direction
):

    score = 0

    timeframes = mtf_data.get(
        "timeframes",
        {}
    )

    weights = {
        "5m": 1,
        "15m": 2,
        "30m": 3,
        "1h": 4
    }

    for interval, weight in weights.items():

        trend = timeframes.get(
            interval,
            "UNKNOWN"
        )

        if direction == "BULLISH":

            if trend == "BULLISH":
                score += weight

            elif trend == "BEARISH":
                score -= weight

        else:

            if trend == "BEARISH":
                score += weight

            elif trend == "BULLISH":
                score -= weight

    return score


# ============================================================
# FALSE BREAKOUT / BREAKDOWN
# ============================================================

def detect_false_breakout(
    df,
    resistance,
    support,
    volume_confirmation
):

    result = {
        "false_breakout": False,
        "false_breakdown": False,
        "breakout_confirmed": False,
        "breakdown_confirmed": False
    }

    if df is None or len(df) < 3:

        return result

    previous = df.iloc[-2]
    current = df.iloc[-1]

    previous_close = safe_float(
        previous["close"]
    )

    current_close = safe_float(
        current["close"]
    )

    current_high = safe_float(
        current["high"]
    )

    current_low = safe_float(
        current["low"]
    )

    if (
        current_close is None or
        previous_close is None
    ):

        return result

    volume_confirmed = bool(
        volume_confirmation.get(
            "confirmed",
            False
        )
    )

    # --------------------------------------------------------
    # Resistance breakout
    # --------------------------------------------------------

    if resistance is not None:

        if current_close > resistance:

            if volume_confirmed:

                result[
                    "breakout_confirmed"
                ] = True

            else:

                # Price is above resistance but
                # volume does not confirm it.
                result[
                    "false_breakout"
                ] = True

        # Previous candle broke out,
        # current candle returned below.
        if (
            previous_close > resistance
            and
            current_close < resistance
        ):

            result[
                "false_breakout"
            ] = True

    # --------------------------------------------------------
    # Support breakdown
    # --------------------------------------------------------

    if support is not None:

        if current_close < support:

            if volume_confirmed:

                result[
                    "breakdown_confirmed"
                ] = True

            else:

                result[
                    "false_breakdown"
                ] = True

        if (
            previous_close < support
            and
            current_close > support
        ):

            result[
                "false_breakdown"
            ] = True

    return result


# ============================================================
# SIGNAL ENGINE V5
# ============================================================

def generate_signal_v5(
    row,
    market_status,
    support,
    resistance,
    support2,
    resistance2,
    mtf_data,
    regime_data,
    opening_range,
    previous_day,
    weekly_levels,
    volume_confirmation,
    breakout_data
):

    close = safe_float(
        row["close"]
    )

    ema9 = safe_float(
        row["ema_9"]
    )

    ema20 = safe_float(
        row["ema_20"]
    )

    ema50 = safe_float(
        row["ema_50"]
    )

    ema100 = safe_float(
        row["ema_100"]
    )

    ema200 = safe_float(
        row["ema_200"]
    )

    rsi = safe_float(
        row["rsi_14"]
    )

    macd = safe_float(
        row["macd"]
    )

    macd_signal = safe_float(
        row["macd_signal"]
    )

    macd_histogram = safe_float(
        row["macd_histogram"]
    )

    momentum = safe_float(
        row["momentum"]
    )

    vwap = safe_float(
        row["vwap"]
    )

    adx = safe_float(
        row["adx_14"]
    )

    atr = safe_float(
        row["atr_14"]
    )

    bollinger_upper = safe_float(
        row["bollinger_upper"]
    )

    bollinger_lower = safe_float(
        row["bollinger_lower"]
    )

    if close is None:

        return {
            "signal": "NEUTRAL",
            "signal_strength": "LOW",
            "confidence": 0,
            "bullish_score": 0,
            "bearish_score": 0,
            "max_score": 100,
            "reasons": [],
            "warnings": [
                "No valid price data"
            ],
            "breakout_confirmed": False,
            "breakdown_confirmed": False,
            "trend": "SIDEWAYS",
            "market_regime":
                "UNKNOWN",
            "signal_quality":
                "LOW",
            "no_trade_reason":
                "No valid price data"
        }

    bullish_score = 0.0
    bearish_score = 0.0

    bullish_reasons = []
    bearish_reasons = []
    warnings = []

    # ========================================================
    # 1. EMA STRUCTURE
    # Weight: 15
    # ========================================================

    if (
        ema9 is not None and
        ema20 is not None
    ):

        if ema9 > ema20:

            bullish_score += 5

            bullish_reasons.append(
                "EMA9 above EMA20"
            )

        elif ema9 < ema20:

            bearish_score += 5

            bearish_reasons.append(
                "EMA9 below EMA20"
            )

    if (
        ema20 is not None and
        ema50 is not None
    ):

        if ema20 > ema50:

            bullish_score += 5

            bullish_reasons.append(
                "EMA20 above EMA50"
            )

        elif ema20 < ema50:

            bearish_score += 5

            bearish_reasons.append(
                "EMA20 below EMA50"
            )

    if (
        ema50 is not None and
        ema100 is not None
    ):

        if ema50 > ema100:

            bullish_score += 3

        elif ema50 < ema100:

            bearish_score += 3

    if (
        ema100 is not None and
        ema200 is not None
    ):

        if ema100 > ema200:

            bullish_score += 2

        elif ema100 < ema200:

            bearish_score += 2

    # ========================================================
    # 2. RSI
    # Weight: 10
    # ========================================================

    if rsi is not None:

        if rsi >= 60:

            bullish_score += 10

            bullish_reasons.append(
                "RSI strong bullish"
            )

        elif rsi >= 55:

            bullish_score += 6

            bullish_reasons.append(
                "RSI bullish"
            )

        elif rsi <= 40:

            bearish_score += 10

            bearish_reasons.append(
                "RSI strong bearish"
            )

        elif rsi <= 45:

            bearish_score += 6

            bearish_reasons.append(
                "RSI bearish"
            )

    # ========================================================
    # 3. MACD
    # Weight: 10
    # ========================================================

    if (
        macd is not None and
        macd_signal is not None
    ):

        if macd > macd_signal:

            bullish_score += 6

            bullish_reasons.append(
                "MACD bullish"
            )

        elif macd < macd_signal:

            bearish_score += 6

            bearish_reasons.append(
                "MACD bearish"
            )

    if macd_histogram is not None:

        if macd_histogram > 0:

            bullish_score += 4

        elif macd_histogram < 0:

            bearish_score += 4

    # ========================================================
    # 4. VWAP
    # Weight: 10
    # ========================================================

    if vwap is not None:

        if close > vwap:

            bullish_score += 10

            bullish_reasons.append(
                "Price above VWAP"
            )

        elif close < vwap:

            bearish_score += 10

            bearish_reasons.append(
                "Price below VWAP"
            )

    # ========================================================
    # 5. MOMENTUM
    # Weight: 5
    # ========================================================

    if momentum is not None:

        if momentum > 0:

            bullish_score += 5

            bullish_reasons.append(
                "Positive momentum"
            )

        elif momentum < 0:

            bearish_score += 5

            bearish_reasons.append(
                "Negative momentum"
            )

    # ========================================================
    # 6. ADX / TREND STRENGTH
    # Weight: 10
    # ========================================================

    strong_trend = False

    if adx is not None:

        if adx >= 30:

            strong_trend = True

            if ema20 is not None:

                if close > ema20:

                    bullish_score += 10

                    bullish_reasons.append(
                        "Strong bullish trend"
                    )

                elif close < ema20:

                    bearish_score += 10

                    bearish_reasons.append(
                        "Strong bearish trend"
                    )

        elif adx >= 25:

            if ema20 is not None:

                if close > ema20:

                    bullish_score += 5

                elif close < ema20:

                    bearish_score += 5

        else:

            warnings.append(
                "ADX weak - trend strength low"
            )

    # ========================================================
    # 7. MULTI-TIMEFRAME
    # Weight: 15
    # ========================================================

    mtf_bull_score = calculate_mtf_score(
        mtf_data,
        "BULLISH"
    )

    mtf_bear_score = calculate_mtf_score(
        mtf_data,
        "BEARISH"
    )

    if mtf_bull_score > 0:

        mtf_points = min(
            15,
            mtf_bull_score * 1.5
        )

        bullish_score += mtf_points

        if mtf_bull_score >= 6:

            bullish_reasons.append(
                "Multiple timeframes bullish"
            )

    elif mtf_bull_score < 0:

        bearish_penalty = min(
            8,
            abs(mtf_bull_score)
        )

        bearish_score += bearish_penalty

    if mtf_bear_score > 0:

        mtf_points = min(
            15,
            mtf_bear_score * 1.5
        )

        bearish_score += mtf_points

        if mtf_bear_score >= 6:

            bearish_reasons.append(
                "Multiple timeframes bearish"
            )

    # ========================================================
    # 8. VOLUME
    # Weight: 10
    # ========================================================

    if volume_confirmation.get(
        "confirmed",
        False
    ):

        direction = volume_confirmation.get(
            "direction",
            ""
        )

        if direction == "BULLISH":

            bullish_score += 10

            bullish_reasons.append(
                "Volume confirms bullish move"
            )

        elif direction == "BEARISH":

            bearish_score += 10

            bearish_reasons.append(
                "Volume confirms bearish move"
            )

    else:

        if volume_confirmation.get(
            "status"
        ) == "LOW":

            warnings.append(
                "Low volume - confirmation weak"
            )

    # ========================================================
    # 9. OPENING RANGE
    # Weight: 10
    # ========================================================

    if opening_range.get(
        "opening_range_breakout",
        False
    ):

        bullish_score += 10

        bullish_reasons.append(
            "Opening range breakout"
        )

    if opening_range.get(
        "opening_range_breakdown",
        False
    ):

        bearish_score += 10

        bearish_reasons.append(
            "Opening range breakdown"
        )

    # ========================================================
    # 10. SUPPORT / RESISTANCE
    # Weight: 10
    # ========================================================

    breakout_confirmed = bool(
        breakout_data.get(
            "breakout_confirmed",
            False
        )
    )

    breakdown_confirmed = bool(
        breakout_data.get(
            "breakdown_confirmed",
            False
        )
    )

    false_breakout = bool(
        breakout_data.get(
            "false_breakout",
            False
        )
    )

    false_breakdown = bool(
        breakout_data.get(
            "false_breakdown",
            False
        )
    )

    if breakout_confirmed:

        bullish_score += 10

        bullish_reasons.append(
            "Confirmed resistance breakout"
        )

    elif false_breakout:

        bearish_score += 7

        bearish_reasons.append(
            "False breakout detected"
        )

        warnings.append(
            "Resistance breakout not confirmed"
        )

    if breakdown_confirmed:

        bearish_score += 10

        bearish_reasons.append(
            "Confirmed support breakdown"
        )

    elif false_breakdown:

        bullish_score += 7

        bullish_reasons.append(
            "False breakdown detected"
        )

        warnings.append(
            "Support breakdown not confirmed"
        )

    # ========================================================
    # 11. PREVIOUS DAY LEVELS
    # ========================================================

    previous_high = safe_float(
        previous_day.get(
            "previous_day_high"
        )
    )

    previous_low = safe_float(
        previous_day.get(
            "previous_day_low"
        )
    )

    previous_close = safe_float(
        previous_day.get(
            "previous_day_close"
        )
    )

    if previous_high is not None:

        if close > previous_high:

            bullish_score += 5

            bullish_reasons.append(
                "Above previous day high"
            )

    if previous_low is not None:

        if close < previous_low:

            bearish_score += 5

            bearish_reasons.append(
                "Below previous day low"
            )

    if previous_close is not None:

        if close > previous_close:

            bullish_score += 2

        elif close < previous_close:

            bearish_score += 2

    # ========================================================
    # 12. WEEKLY LEVELS
    # ========================================================

    weekly_high = safe_float(
        weekly_levels.get(
            "weekly_high"
        )
    )

    weekly_low = safe_float(
        weekly_levels.get(
            "weekly_low"
        )
    )

    if weekly_high is not None:

        if close > weekly_high:

            bullish_score += 3

    if weekly_low is not None:

        if close < weekly_low:

            bearish_score += 3

    # ========================================================
    # 13. MARKET REGIME
    # ========================================================

    market_regime = regime_data.get(
        "market_regime",
        "UNKNOWN"
    )

    if market_regime == "TRENDING BULLISH":

        bullish_score += 5

        bullish_reasons.append(
            "Bullish market regime"
        )

    elif market_regime == "TRENDING BEARISH":

        bearish_score += 5

        bearish_reasons.append(
            "Bearish market regime"
        )

    elif market_regime == "SIDEWAYS":

        warnings.append(
            "Sideways market - signal quality reduced"
        )

    elif market_regime == "LOW VOLATILITY":

        warnings.append(
            "Low volatility - breakout risk"
        )

    elif market_regime == "HIGH VOLATILITY":

        warnings.append(
            "High volatility - risk increased"
        )

    # ========================================================
    # SCORE NORMALIZATION
    # ========================================================

    raw_difference = (
        bullish_score -
        bearish_score
    )

    total_score = (
        bullish_score +
        bearish_score
    )

    # We use a 100-point style score.
    max_possible = 100.0

    bullish_percent = clamp(
        (
            bullish_score /
            max(
                total_score,
                1
            )
        ) * 100,
        0,
        100
    )

    bearish_percent = clamp(
        (
            bearish_score /
            max(
                total_score,
                1
            )
        ) * 100,
        0,
        100
    )

    # ========================================================
    # TREND
    # ========================================================

    trend = calculate_trend(
        row
    )

    # ========================================================
    # SIGNAL
    # ========================================================

    signal = "NEUTRAL"
    strength = "LOW"

    signal_quality = "LOW"

    no_trade_reason = None

    # The system should not trade if:
    # - market is not live
    # - signal conflict is too high
    # - sideways market
    # - weak score difference
    # - false breakout
    # - insufficient confirmation

    difference = abs(
        bullish_score -
        bearish_score
    )

    mtf_overall = mtf_data.get(
        "overall",
        "UNKNOWN"
    )

    if market_status != "LIVE":

        no_trade_reason = (
            f"Market status is {market_status}"
        )

    elif false_breakout:

        no_trade_reason = (
            "False breakout detected"
        )

    elif false_breakdown:

        no_trade_reason = (
            "False breakdown detected"
        )

    elif market_regime == "SIDEWAYS" and difference < 12:

        no_trade_reason = (
            "Sideways market with weak edge"
        )

    elif total_score < 25:

        no_trade_reason = (
            "Insufficient confirmation"
        )

    elif difference < 10:

        no_trade_reason = (
            "Bullish and bearish factors are too close"
        )

    else:

        if (
            bullish_score > bearish_score
            and
            bullish_score >= 45
            and
            difference >= 10
        ):

            signal = "BUY"

            if (
                bullish_score >= 70
                and
                difference >= 25
            ):

                strength = "HIGH"

            elif (
                bullish_score >= 55
                and
                difference >= 15
            ):

                strength = "MEDIUM"

            else:

                strength = "LOW"

        elif (
            bearish_score > bullish_score
            and
            bearish_score >= 45
            and
            difference >= 10
        ):

            signal = "SELL"

            if (
                bearish_score >= 70
                and
                difference >= 25
            ):

                strength = "HIGH"

            elif (
                bearish_score >= 55
                and
                difference >= 15
            ):

                strength = "MEDIUM"

            else:

                strength = "LOW"

        else:

            no_trade_reason = (
                "Signal strength is insufficient"
            )

    # ========================================================
    # CONFIDENCE
    # ========================================================

    if total_score <= 0:

        confidence = 50.0

    else:

        confidence = max(
            bullish_percent,
            bearish_percent
        )

    # Penalize weak confirmation.

    if market_regime == "SIDEWAYS":

        confidence -= 8

    if market_regime == "LOW VOLATILITY":

        confidence -= 5

    if mtf_overall == "MIXED":

        confidence -= 7

    if not volume_confirmation.get(
        "confirmed",
        False
    ):

        confidence -= 3

    if false_breakout or false_breakdown:

        confidence -= 15

    confidence = round(
        clamp(
            confidence,
            0,
            100
        ),
        1
    )

    # ========================================================
    # QUALITY
    # ========================================================

    if (
        confidence >= 80
        and
        difference >= 25
    ):

        signal_quality = "EXCELLENT"

    elif (
        confidence >= 70
        and
        difference >= 18
    ):

        signal_quality = "GOOD"

    elif (
        confidence >= 60
        and
        difference >= 12
    ):

        signal_quality = "MODERATE"

    else:

        signal_quality = "LOW"

    # ========================================================
    # REASONS
    # ========================================================

    if signal == "BUY":

        reasons = bullish_reasons

    elif signal == "SELL":

        reasons = bearish_reasons

    elif bullish_score > bearish_score:

        reasons = bullish_reasons

    elif bearish_score > bullish_score:

        reasons = bearish_reasons

    else:

        reasons = [
            "Market factors are balanced"
        ]

    if not reasons:

        reasons = [
            "No strong directional confirmation"
        ]

    if no_trade_reason is not None:

        warnings.append(
            no_trade_reason
        )

    return {

        "signal":
            signal,

        "signal_strength":
            strength,

        "confidence":
            confidence,

        "bullish_score":
            round(
                bullish_score,
                1
            ),

        "bearish_score":
            round(
                bearish_score,
                1
            ),

        "max_score":
            max_possible,

        "bullish_percent":
            round(
                bullish_percent,
                1
            ),

        "bearish_percent":
            round(
                bearish_percent,
                1
            ),

        "reasons":
            reasons,

        "warnings":
            list(dict.fromkeys(
                warnings
            )),

        "breakout_confirmed":
            breakout_confirmed,

        "breakdown_confirmed":
            breakdown_confirmed,

        "false_breakout":
            false_breakout,

        "false_breakdown":
            false_breakdown,

        "trend":
            trend,

        "market_regime":
            market_regime,

        "regime_strength":
            regime_data.get(
                "regime_strength",
                "LOW"
            ),

        "signal_quality":
            signal_quality,

        "no_trade_reason":
            no_trade_reason,

        "strong_trend":
            strong_trend,

        "atr":
            atr
    }


# ============================================================
# TRADE PLAN V5
# ============================================================

def create_trade_plan_v5(
    signal_data,
    row,
    support,
    resistance,
    previous_day,
    opening_range
):

    signal = signal_data[
        "signal"
    ]

    close = safe_float(
        row["close"]
    )

    atr = safe_float(
        row["atr_14"]
    )

    if (
        signal not in (
            "BUY",
            "SELL"
        )
        or
        close is None
        or
        atr is None
        or
        atr <= 0
    ):

        return {

            "trade_status":
                "NO TRADE",

            "entry":
                None,

            "stop_loss":
                None,

            "target_1":
                None,

            "target_2":
                None,

            "risk_points":
                None,

            "reward_1_points":
                None,

            "reward_2_points":
                None,

            "risk_reward_1":
                None,

            "risk_reward_2":
                None,

            "trailing_stop":
                None,

            "entry_zone":
                None,

            "avoid_zone":
                None
        }

    # --------------------------------------------------------
    # ATR risk model
    # --------------------------------------------------------

    stop_multiplier = 1.25

    target1_multiplier = 1.8

    target2_multiplier = 2.8

    if signal_data.get(
        "market_regime"
    ) == "HIGH VOLATILITY":

        stop_multiplier = 1.5

        target1_multiplier = 2.0

        target2_multiplier = 3.2

    # --------------------------------------------------------
    # BUY
    # --------------------------------------------------------

    if signal == "BUY":

        entry = close

        atr_stop = (
            entry -
            (
                atr *
                stop_multiplier
            )
        )

        structural_stop = None

        if support is not None:

            structural_stop = (
                support -
                (
                    atr * 0.20
                )
            )

        if structural_stop is not None:

            stop_loss = min(
                atr_stop,
                structural_stop
            )

        else:

            stop_loss = atr_stop

        target1 = (
            entry +
            (
                atr *
                target1_multiplier
            )
        )

        target2 = (
            entry +
            (
                atr *
                target2_multiplier
            )
        )

        # If resistance is close,
        # don't place target1 below it.

        if (
            resistance is not None
            and
            resistance > entry
        ):

            target1 = max(
                target1,
                resistance
            )

    # --------------------------------------------------------
    # SELL
    # --------------------------------------------------------

    else:

        entry = close

        atr_stop = (
            entry +
            (
                atr *
                stop_multiplier
            )
        )

        structural_stop = None

        if resistance is not None:

            structural_stop = (
                resistance +
                (
                    atr * 0.20
                )
            )

        if structural_stop is not None:

            stop_loss = max(
                atr_stop,
                structural_stop
            )

        else:

            stop_loss = atr_stop

        target1 = (
            entry -
            (
                atr *
                target1_multiplier
            )
        )

        target2 = (
            entry -
            (
                atr *
                target2_multiplier
            )
        )

        if (
            support is not None
            and
            support < entry
        ):

            target1 = min(
                target1,
                support
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
        f"{entry - atr * 0.15:.2f}"
        f" - "
        f"{entry + atr * 0.15:.2f}"
    )

    trailing_stop = (
        atr *
        1.0
    )

    return {

        "trade_status":
            "TRADE SETUP",

        "entry":
            round(
                entry,
                2
            ),

        "stop_loss":
            round(
                stop_loss,
                2
            ),

        "target_1":
            round(
                target1,
                2
            ),

        "target_2":
            round(
                target2,
                2
            ),

        "risk_points":
            round(
                risk,
                2
            ),

        "reward_1_points":
            round(
                reward1,
                2
            ),

        "reward_2_points":
            round(
                reward2,
                2
            ),

        "risk_reward_1":
            round(
                rr1,
                2
            )
            if rr1 is not None
            else None,

        "risk_reward_2":
            round(
                rr2,
                2
            )
            if rr2 is not None
            else None,

        "trailing_stop":
            round(
                trailing_stop,
                2
            ),

        "entry_zone":
            entry_zone,

        "avoid_zone":
            None
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def build_analysis(interval="5m"):

    # --------------------------------------------------------
    # Main timeframe
    # --------------------------------------------------------

    df = fetch_history(
        interval
    )

    df = calculate_indicators(
        df
    )

    row = df.iloc[-1]

    market_info = calculate_market_status(
        df
    )

    market_status = market_info[
        "market_status"
    ]

    # --------------------------------------------------------
    # Support / resistance
    # --------------------------------------------------------

    (
        support,
        support2,
        resistance,
        resistance2
    ) = calculate_support_resistance(
        df
    )

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    (
        volume_status,
        volume_ratio
    ) = calculate_volume_status(
        df
    )

    # --------------------------------------------------------
    # Preliminary volume direction
    # --------------------------------------------------------

    preliminary_direction = "BULLISH"

    current_trend = calculate_trend(
        row
    )

    if current_trend == "BEARISH":

        preliminary_direction = "BEARISH"

    volume_confirmation = (
        calculate_volume_confirmation(
            df,
            preliminary_direction
        )
    )

    volume_confirmation[
        "direction"
    ] = preliminary_direction

    # --------------------------------------------------------
    # Previous day
    # --------------------------------------------------------

    previous_day = (
        calculate_previous_day_levels()
    )

    # --------------------------------------------------------
    # Weekly
    # --------------------------------------------------------

    weekly_levels = (
        calculate_weekly_levels()
    )

    # --------------------------------------------------------
    # Opening range
    # --------------------------------------------------------

    opening_range = (
        calculate_opening_range(
            df
        )
    )

    # --------------------------------------------------------
    # Market regime
    # --------------------------------------------------------

    regime_data = (
        calculate_market_regime(
            row,
            df
        )
    )

    # --------------------------------------------------------
    # Multi timeframe
    # --------------------------------------------------------

    mtf_data = (
        calculate_multi_timeframe()
    )

    # --------------------------------------------------------
    # False breakout / breakdown
    # --------------------------------------------------------

    breakout_data = (
        detect_false_breakout(
            df,
            resistance,
            support,
            volume_confirmation
        )
    )

    # --------------------------------------------------------
    # Signal
    # --------------------------------------------------------

    signal_data = (
        generate_signal_v5(

            row=row,

            market_status=
                market_status,

            support=
                support,

            resistance=
                resistance,

            support2=
                support2,

            resistance2=
                resistance2,

            mtf_data=
                mtf_data,

            regime_data=
                regime_data,

            opening_range=
                opening_range,

            previous_day=
                previous_day,

            weekly_levels=
                weekly_levels,

            volume_confirmation=
                volume_confirmation,

            breakout_data=
                breakout_data
        )
    )

    # --------------------------------------------------------
    # Trade plan
    # --------------------------------------------------------

    trade_plan = (
        create_trade_plan_v5(

            signal_data=
                signal_data,

            row=
                row,

            support=
                support,

            resistance=
                resistance,

            previous_day=
                previous_day,

            opening_range=
                opening_range
        )
    )

    # --------------------------------------------------------
    # Price
    # --------------------------------------------------------

    price = safe_float(
        row["close"]
    )

    previous_close = None

    if len(df) >= 2:

        previous_close = safe_float(
            df["close"].iloc[-2]
        )

    change = None
    change_percent = None

    if (
        price is not None
        and
        previous_close is not None
        and
        previous_close != 0
    ):

        change = (
            price -
            previous_close
        )

        change_percent = (
            change /
            previous_close
        ) * 100

    # --------------------------------------------------------
    # Higher timeframe
    # --------------------------------------------------------

    higher_tf_trend = (
        mtf_data.get(
            "timeframes",
            {}
        ).get(
            "15m",
            "SIDEWAYS"
        )
    )

    trend = signal_data[
        "trend"
    ]

    higher_tf_conflict = (
        trend != "SIDEWAYS"
        and
        higher_tf_trend != "SIDEWAYS"
        and
        trend != higher_tf_trend
    )

    # --------------------------------------------------------
    # Distance to support/resistance
    # --------------------------------------------------------

    support_distance_percent = None

    resistance_distance_percent = None

    if (
        price is not None
        and
        support is not None
        and
        price != 0
    ):

        support_distance_percent = (
            abs(
                price -
                support
            ) /
            price
        ) * 100

    if (
        price is not None
        and
        resistance is not None
        and
        price != 0
    ):

        resistance_distance_percent = (
            abs(
                resistance -
                price
            ) /
            price
        ) * 100

    near_support = (
        support_distance_percent
        is not None
        and
        support_distance_percent
        <= 0.15
    )

    near_resistance = (
        resistance_distance_percent
        is not None
        and
        resistance_distance_percent
        <= 0.15
    )

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

        warnings.append(
            "Volume data unavailable"
        )

    if higher_tf_conflict:

        warnings.append(
            "Higher timeframe trend conflict"
        )

    warnings = list(
        dict.fromkeys(
            warnings
        )
    )

    # --------------------------------------------------------
    # Final response
    # --------------------------------------------------------

    return {

        # ====================================================
        # BASIC
        # ====================================================

        "symbol":
            "NIFTY 50",

        "price":
            round_or_none(
                price
            ),

        "previous_close":
            round_or_none(
                previous_close
            ),

        "change":
            round_or_none(
                change
            ),

        "change_percent":
            round_or_none(
                change_percent
            ),

        "market_status":
            market_status,

        "data_age_seconds":
            market_info[
                "data_age_seconds"
            ],

        "latest_data_time":
            market_info[
                "latest_data_time"
            ],

        # ====================================================
        # TREND / SIGNAL
        # ====================================================

        "trend":
            trend,

        "higher_timeframe_trend":
            higher_tf_trend,

        "higher_tf_conflict":
            higher_tf_conflict,

        "signal":
            signal_data[
                "signal"
            ],

        "signal_strength":
            signal_data[
                "signal_strength"
            ],

        "confidence":
            signal_data[
                "confidence"
            ],

        "bullish_score":
            signal_data[
                "bullish_score"
            ],

        "bearish_score":
            signal_data[
                "bearish_score"
            ],

        "max_score":
            signal_data[
                "max_score"
            ],

        "bullish_percent":
            signal_data[
                "bullish_percent"
            ],

        "bearish_percent":
            signal_data[
                "bearish_percent"
            ],

        "signal_quality":
            signal_data[
                "signal_quality"
            ],

        "no_trade_reason":
            signal_data[
                "no_trade_reason"
            ],

        # ====================================================
        # MULTI TIMEFRAME
        # ====================================================

        "mtf_overall":
            mtf_data[
                "overall"
            ],

        "mtf_bullish_count":
            mtf_data[
                "bullish_count"
            ],

        "mtf_bearish_count":
            mtf_data[
                "bearish_count"
            ],

        "mtf_sideways_count":
            mtf_data[
                "sideways_count"
            ],

        "trend_5m":
            mtf_data[
                "timeframes"
            ].get(
                "5m",
                "UNKNOWN"
            ),

        "trend_15m":
            mtf_data[
                "timeframes"
            ].get(
                "15m",
                "UNKNOWN"
            ),

        "trend_30m":
            mtf_data[
                "timeframes"
            ].get(
                "30m",
                "UNKNOWN"
            ),

        "trend_1h":
            mtf_data[
                "timeframes"
            ].get(
                "1h",
                "UNKNOWN"
            ),

        # ====================================================
        # MARKET REGIME
        # ====================================================

        "market_regime":
            regime_data[
                "market_regime"
            ],

        "regime_strength":
            regime_data[
                "regime_strength"
            ],

        "regime_score":
            regime_data[
                "regime_score"
            ],

        # ====================================================
        # MOVING AVERAGES
        # ====================================================

        "ma_5":
            round_or_none(
                row["ma_5"]
            ),

        "ma_10":
            round_or_none(
                row["ma_10"]
            ),

        "ma_20":
            round_or_none(
                row["ma_20"]
            ),

        "ema_9":
            round_or_none(
                row["ema_9"]
            ),

        "ema_20":
            round_or_none(
                row["ema_20"]
            ),

        "ema_50":
            round_or_none(
                row["ema_50"]
            ),

        "ema_100":
            round_or_none(
                row["ema_100"]
            ),

        "ema_200":
            round_or_none(
                row["ema_200"]
            ),

        # ====================================================
        # INDICATORS
        # ====================================================

        "rsi_14":
            round_or_none(
                row["rsi_14"]
            ),

        "macd":
            round_or_none(
                row["macd"]
            ),

        "macd_signal":
            round_or_none(
                row["macd_signal"]
            ),

        "macd_histogram":
            round_or_none(
                row["macd_histogram"]
            ),

        "momentum":
            round_or_none(
                row["momentum"]
            ),

        "atr_14":
            round_or_none(
                row["atr_14"]
            ),

        "adx_14":
            round_or_none(
                row["adx_14"]
            ),

        "vwap":
            round_or_none(
                row["vwap"]
            ),

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

        "bollinger_width":
            round_or_none(
                row["bollinger_width"],
                3
            ),

        # ====================================================
        # SUPPORT / RESISTANCE
        # ====================================================

        "support":
            round_or_none(
                support
            ),

        "support_2":
            round_or_none(
                support2
            ),

        "resistance":
            round_or_none(
                resistance
            ),

        "resistance_2":
            round_or_none(
                resistance2
            ),

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

        # ====================================================
        # PREVIOUS DAY
        # ====================================================

        "previous_day_high":
            round_or_none(
                previous_day[
                    "previous_day_high"
                ]
            ),

        "previous_day_low":
            round_or_none(
                previous_day[
                    "previous_day_low"
                ]
            ),

        "previous_day_close":
            round_or_none(
                previous_day[
                    "previous_day_close"
                ]
            ),

        # ====================================================
        # WEEKLY
        # ====================================================

        "weekly_high":
            round_or_none(
                weekly_levels[
                    "weekly_high"
                ]
            ),

        "weekly_low":
            round_or_none(
                weekly_levels[
                    "weekly_low"
                ]
            ),

        # ====================================================
        # OPENING RANGE
        # ====================================================

        "opening_range_high":
            round_or_none(
                opening_range[
                    "opening_range_high"
                ]
            ),

        "opening_range_low":
            round_or_none(
                opening_range[
                    "opening_range_low"
                ]
            ),

        "opening_range_breakout":
            opening_range[
                "opening_range_breakout"
            ],

        "opening_range_breakdown":
            opening_range[
                "opening_range_breakdown"
            ],

        # ====================================================
        # BREAKOUT
        # ====================================================

        "breakout_confirmed":
            signal_data[
                "breakout_confirmed"
            ],

        "breakdown_confirmed":
            signal_data[
                "breakdown_confirmed"
            ],

        "false_breakout":
            signal_data[
                "false_breakout"
            ],

        "false_breakdown":
            signal_data[
                "false_breakdown"
            ],

        # ====================================================
        # VOLUME
        # ====================================================

        "volume_status":
            volume_status,

        "volume_ratio":
            volume_ratio,

        "volume_confirmation":
            volume_confirmation[
                "confirmed"
            ],

        "volume_confirmation_status":
            volume_confirmation[
                "status"
            ],

        "volume_confirmation_reason":
            volume_confirmation[
                "reason"
            ],

        # ====================================================
        # TRADE PLAN
        # ====================================================

        "entry_zone":
            trade_plan[
                "entry_zone"
            ],

        "avoid_zone":
            trade_plan[
                "avoid_zone"
            ],

        "trade_status":
            trade_plan[
                "trade_status"
            ],

        "entry":
            trade_plan[
                "entry"
            ],

        "stop_loss":
            trade_plan[
                "stop_loss"
            ],

        "target_1":
            trade_plan[
                "target_1"
            ],

        "target_2":
            trade_plan[
                "target_2"
            ],

        "risk_points":
            trade_plan[
                "risk_points"
            ],

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

        # ====================================================
        # EXPLANATION
        # ====================================================

        "reasons":
            signal_data[
                "reasons"
            ],

        "warnings":
            warnings,

        # ====================================================
        # VERSION
        # ====================================================

        "analysis_version":
            "V5.0",

        "time":
            datetime.now(
                timezone.utc
            ).isoformat()
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {

        "app":
            "Nifty AI Trader Backend",

        "version":
            "5.0",

        "status":
            "running",

        "symbol":
            "NIFTY 50",

        "data_source":
            "Yahoo Finance",

        "live_detection":
            True,

        "features":
            [
                "Multi-Timeframe Analysis",
                "Market Regime Detection",
                "Previous Day Levels",
                "Weekly Levels",
                "Opening Range",
                "Volume Confirmation",
                "False Breakout Detection",
                "Weighted Signal Engine",
                "ATR Risk Management",
                "NO TRADE Filter"
            ],

        "history_endpoint":
            "/nifty/history?interval=5m",

        "message":
            "Nifty AI Trader V5.0 backend is running."
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {

        "status":
            "ok",

        "version":
            "5.0",

        "time":
            datetime.now(
                timezone.utc
            ).isoformat()
    }


# ============================================================
# NIFTY MAIN
# ============================================================

@app.get("/nifty")
def nifty():

    try:

        return build_analysis(
            "5m"
        )

    except Exception as exc:

        raise HTTPException(
            status_code=503,
            detail=str(exc)
        )


# ============================================================
# HISTORY ENDPOINT
#
# IMPORTANT:
# THIS MUST COME BEFORE /nifty/{interval}
# ============================================================

@app.get("/nifty/history")
def nifty_history(
    interval: str = "5m"
):

    if interval not in ALLOWED_INTERVALS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid interval. "
                "Use 1m, 5m, 15m, "
                "30m, 1h or 1D."
            )
        )

    try:

        df = fetch_history(
            interval
        )

        df = df.tail(
            300
        )

        candles = []

        for index, row in df.iterrows():

            try:

                timestamp = index

                if isinstance(
                    timestamp,
                    pd.Timestamp
                ):

                    if timestamp.tzinfo is None:

                        timestamp = (
                            timestamp.tz_localize(
                                "UTC"
                            )
                        )

                    timestamp = (
                        timestamp.tz_convert(
                            IST
                        )
                    )

                    timestamp_string = (
                        timestamp.isoformat()
                    )

                elif isinstance(
                    timestamp,
                    datetime
                ):

                    if timestamp.tzinfo is None:

                        timestamp = (
                            timestamp.replace(
                                tzinfo=timezone.utc
                            )
                        )

                    timestamp_string = (
                        timestamp.astimezone(
                            IST
                        ).isoformat()
                    )

                else:

                    timestamp_string = str(
                        timestamp
                    )

            except Exception:

                timestamp_string = str(
                    index
                )

            open_price = safe_float(
                row["open"]
            )

            high_price = safe_float(
                row["high"]
            )

            low_price = safe_float(
                row["low"]
            )

            close_price = safe_float(
                row["close"]
            )

            volume_value = safe_float(
                row["volume"]
            )

            if (
                open_price is None
                or
                high_price is None
                or
                low_price is None
                or
                close_price is None
            ):

                continue

            candles.append({

                "timestamp":
                    timestamp_string,

                "open":
                    round(
                        open_price,
                        2
                    ),

                "high":
                    round(
                        high_price,
                        2
                    ),

                "low":
                    round(
                        low_price,
                        2
                    ),

                "close":
                    round(
                        close_price,
                        2
                    ),

                "volume":
                    round(
                        volume_value,
                        2
                    )
                    if volume_value is not None
                    else None
            })

        status = (
            calculate_market_status(
                df
            )
        )

        return {

            "symbol":
                "NIFTY 50",

            "interval":
                interval,

            "market_status":
                status[
                    "market_status"
                ],

            "data_age_seconds":
                status[
                    "data_age_seconds"
                ],

            "latest_data_time":
                status[
                    "latest_data_time"
                ],

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
# INTERVAL ANALYSIS
#
# IMPORTANT:
# THIS COMES AFTER /nifty/history
# ============================================================

@app.get("/nifty/{interval}")
def nifty_interval(
    interval: str
):

    if interval not in ALLOWED_INTERVALS:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid interval. "
                "Use 1m, 5m, 15m, "
                "30m, 1h or 1D."
            )
        )

    try:

        return build_analysis(
            interval
        )

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
