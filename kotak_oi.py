# ============================================================
# NIFTY AI TRADER — KOTAK NEO OI INTELLIGENCE ENGINE
# Version: OI-2.1
#
# Market Data Only
# NO ORDER PLACEMENT
#
# Compatible with Kotak Neo Python SDK 3.0.6
#
# OI INTELLIGENCE:
#   CALL BUYING
#   CALL WRITING
#   PUT BUYING
#   PUT WRITING
#   OI SHIFT
#   OI SCORE
#   OI CONFIDENCE
#   OI BIAS
#   OI SUPPORT / RESISTANCE
# ============================================================

import os
import math
import threading
import time

from datetime import datetime
from typing import Optional

from neo_api_client import NeoAPI


# ============================================================
# CONFIG
# ============================================================

KOTAK_EXCHANGE = "nse_fo"
KOTAK_UNDERLYING = "NIFTY"

DEFAULT_COUNT = 40

OI_CACHE_TTL = 15

# Premium/OI intelligence thresholds
PREMIUM_CHANGE_THRESHOLD = 0.0015
OI_CHANGE_THRESHOLD = 0.02
VOLUME_RATIO_THRESHOLD = 1.20

# Score limits
OI_SCORE_MIN = -10
OI_SCORE_MAX = 10


# ============================================================
# CACHE
# ============================================================

_oi_cache = {}
_oi_cache_lock = threading.Lock()

# Previous option snapshot.
#
# Key:
#   expiry:CE:strike
#   expiry:PE:strike
#
# Value:
#   {
#       "ltp": ...,
#       "oi": ...,
#       "volume": ...,
#       "timestamp": ...
#   }
#
_oi_previous_snapshot = {}

_oi_snapshot_lock = threading.Lock()


# ============================================================
# SAFE HELPERS
# ============================================================

def safe_float(value, digits=2):

    try:

        if value is None:
            return None

        value = float(value)

        if not math.isfinite(value):
            return None

        return round(value, digits)

    except Exception:

        return None


def safe_int(value):

    try:

        if value is None:
            return None

        return int(float(value))

    except Exception:

        return None


def get_env(name: str):

    value = os.getenv(name)

    if value is None:
        return ""

    return value.strip()


def clamp(
    value,
    minimum,
    maximum
):

    return max(
        minimum,
        min(
            maximum,
            value
        )
    )


# ============================================================
# KOTAK CLIENT
# ============================================================

def get_kotak_client():

    consumer_key = get_env(
        "NEO_CONSUMER_KEY"
    )

    if not consumer_key:

        raise RuntimeError(
            "NEO_CONSUMER_KEY is not configured."
        )

    return NeoAPI(
        consumer_key=consumer_key,
        environment="prod"
    )


# ============================================================
# EXPIRY
# ============================================================

def get_nifty_expiries():

    client = get_kotak_client()

    response = client.expiries(
        exchange=KOTAK_EXCHANGE,
        underlying=KOTAK_UNDERLYING,
        instrument_type="option"
    )

    if not isinstance(response, dict):

        raise RuntimeError(
            "Invalid Kotak expiry response."
        )

    expiries = response.get(
        "expiries",
        []
    )

    if not isinstance(expiries, list):

        expiries = []

    today = datetime.now().date()

    valid = []

    for expiry in expiries:

        try:

            expiry_date = datetime.strptime(
                str(expiry),
                "%Y-%m-%d"
            ).date()

            if expiry_date >= today:

                valid.append(
                    str(expiry)
                )

        except Exception:

            continue

    valid.sort()

    return valid


def get_nearest_expiry():

    expiries = get_nifty_expiries()

    if not expiries:

        raise RuntimeError(
            "No valid NIFTY option expiry found."
        )

    return expiries[0]


# ============================================================
# CACHE
# ============================================================

def get_cached_oi(key):

    with _oi_cache_lock:

        item = _oi_cache.get(key)

        if item is None:
            return None

        timestamp, value = item

        if (
            time.time() -
            timestamp
        ) > OI_CACHE_TTL:

            return None

        return value


def set_cached_oi(key, value):

    with _oi_cache_lock:

        _oi_cache[key] = (
            time.time(),
            value
        )


# ============================================================
# OPTION CHAIN CHECK
# ============================================================

def has_option_chain_data(response):

    if not isinstance(response, dict):

        return False

    calls = response.get(
        "call",
        []
    )

    puts = response.get(
        "put",
        []
    )

    if not isinstance(calls, list):
        calls = []

    if not isinstance(puts, list):
        puts = []

    return (
        len(calls) > 0 or
        len(puts) > 0
    )


# ============================================================
# RAW OPTION CHAIN
# ============================================================

def fetch_option_chain(
    expiry: Optional[str] = None,
    count: int = DEFAULT_COUNT
):

    requested_expiry = expiry

    if expiry is None:

        expiry = get_nearest_expiry()

    try:

        count = int(count)

    except Exception:

        count = DEFAULT_COUNT

    count = max(
        10,
        min(count, 100)
    )

    count = (
        count // 10
    ) * 10

    cache_key = (
        f"{expiry}:{count}"
    )

    cached = get_cached_oi(
        cache_key
    )

    if cached is not None:

        return cached

    client = get_kotak_client()

    response = client.option_chain(
        exchange=KOTAK_EXCHANGE,
        underlying=KOTAK_UNDERLYING,
        expiry=expiry,
        instrument_type="option",
        count=count
    )

    if not isinstance(response, dict):

        raise RuntimeError(
            "Invalid Kotak option-chain response."
        )

    response["_requested_expiry"] = (
        requested_expiry
    )

    response["_actual_expiry"] = (
        expiry
    )

    response["_chain_attempt"] = (
        "explicit_expiry"
    )

    if has_option_chain_data(response):

        set_cached_oi(
            cache_key,
            response
        )

        return response

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    try:

        fallback_response = client.option_chain(
            exchange=KOTAK_EXCHANGE,
            underlying=KOTAK_UNDERLYING,
            expiry=None,
            instrument_type="option",
            count=count
        )

        if (
            isinstance(
                fallback_response,
                dict
            )
            and
            has_option_chain_data(
                fallback_response
            )
        ):

            fallback_response[
                "_requested_expiry"
            ] = requested_expiry

            fallback_response[
                "_actual_expiry"
            ] = expiry

            fallback_response[
                "_chain_attempt"
            ] = (
                "nearest_expiry_fallback"
            )

            set_cached_oi(
                cache_key,
                fallback_response
            )

            return fallback_response

    except Exception:

        pass

    response[
        "_chain_attempt"
    ] = "no_data"

    return response


# ============================================================
# OPTION NORMALIZER
# ============================================================

def normalize_option(
    item,
    option_type: str
):

    if not isinstance(item, dict):

        return None

    inst = item.get(
        "inst",
        {}
    )

    quote = item.get(
        "quote",
        {}
    )

    oi = item.get(
        "oi",
        {}
    )

    if not isinstance(inst, dict):
        inst = {}

    if not isinstance(quote, dict):
        quote = {}

    if not isinstance(oi, dict):
        oi = {}

    strike = safe_float(
        inst.get(
            "strkPrc"
        )
    )

    if strike is None:

        return None

    current_oi = safe_int(
        oi.get(
            "cur"
        )
    )

    previous_oi = safe_int(
        oi.get(
            "prev"
        )
    )

    change_oi = safe_float(
        oi.get(
            "chg"
        ),
        2
    )

    change_pct = safe_float(
        oi.get(
            "chgPct"
        ),
        2
    )

    ltp = safe_float(
        quote.get(
            "ltp"
        )
    )

    volume = safe_int(
        quote.get(
            "vol"
        )
    )

    return {

        "option_type": option_type,

        "strike": strike,

        "symbol": inst.get(
            "symbol"
        ),

        "neo_symbol": inst.get(
            "neoSymbol"
        ),

        "moneyness": inst.get(
            "moneyness"
        ),

        "ltp": ltp,

        "volume": volume,

        "oi": current_oi,

        "previous_oi": previous_oi,

        "change_oi": change_oi,

        "change_oi_percent": change_pct
    }


# ============================================================
# NORMALIZE COMPLETE CHAIN
# ============================================================

def normalize_chain(
    raw_response: dict,
    expiry: str
):

    calls_raw = raw_response.get(
        "call",
        []
    )

    puts_raw = raw_response.get(
        "put",
        []
    )

    if not isinstance(calls_raw, list):
        calls_raw = []

    if not isinstance(puts_raw, list):
        puts_raw = []

    calls = []

    puts = []

    for item in calls_raw:

        row = normalize_option(
            item,
            "CE"
        )

        if row is not None:

            calls.append(row)

    for item in puts_raw:

        row = normalize_option(
            item,
            "PE"
        )

        if row is not None:

            puts.append(row)

    calls.sort(
        key=lambda x: x["strike"]
    )

    puts.sort(
        key=lambda x: x["strike"]
    )

    common = raw_response.get(
        "common_data",
        {}
    )

    if not isinstance(common, dict):
        common = {}

    spot = raw_response.get(
        "spot",
        {}
    )

    if not isinstance(spot, dict):
        spot = {}

    future = raw_response.get(
        "future",
        {}
    )

    if not isinstance(future, dict):
        future = {}

    return {

        "expiry": expiry,

        "underlying": KOTAK_UNDERLYING,

        "exchange": KOTAK_EXCHANGE,

        "lot_size": safe_int(
            common.get(
                "mktLot"
            )
        ),

        "multiplier": safe_int(
            common.get(
                "multiplier"
            )
        ),

        "calls": calls,

        "puts": puts,

        "spot": {

            "symbol": spot.get(
                "symbol"
            ),

            "ltp": safe_float(
                spot.get(
                    "ltp"
                )
            ),

            "previous_close": safe_float(
                spot.get(
                    "prevClose"
                )
            )
        },

        "future": {

            "symbol": future.get(
                "symbol"
            ),

            "ltp": safe_float(
                future.get(
                    "ltp"
                )
            ),

            "expiry": future.get(
                "expiry"
            )
        }
    }


# ============================================================
# PREVIOUS OPTION SNAPSHOT
# ============================================================

def get_previous_option_snapshot(
    expiry,
    option_type,
    strike
):

    key = (
        f"{expiry}:"
        f"{option_type}:"
        f"{strike}"
    )

    with _oi_snapshot_lock:

        return _oi_previous_snapshot.get(
            key
        )


def save_option_snapshot(
    expiry,
    option_type,
    strike,
    ltp,
    oi,
    volume
):

    key = (
        f"{expiry}:"
        f"{option_type}:"
        f"{strike}"
    )

    with _oi_snapshot_lock:

        _oi_previous_snapshot[key] = {

            "ltp": ltp,

            "oi": oi,

            "volume": volume,

            "timestamp": time.time()
        }


# ============================================================
# OPTION PREMIUM CHANGE
# ============================================================

def calculate_premium_change(
    current_ltp,
    previous_ltp
):

    if (
        current_ltp is None
        or
        previous_ltp is None
        or
        previous_ltp <= 0
    ):

        return None

    return safe_float(
        (
            current_ltp -
            previous_ltp
        )
        /
        previous_ltp,
        5
    )


# ============================================================
# OI CHANGE %
# ============================================================

def calculate_oi_change_ratio(
    current_oi,
    previous_oi,
    reported_change_pct=None
):

    if (
        previous_oi is not None
        and
        previous_oi > 0
        and
        current_oi is not None
    ):

        return safe_float(
            (
                current_oi -
                previous_oi
            )
            /
            previous_oi,
            5
        )

    if reported_change_pct is not None:

        return safe_float(
            reported_change_pct / 100.0,
            5
        )

    return None


# ============================================================
# FOUR-WAY OPTION INTELLIGENCE
#
# IMPORTANT:
# These are indications, NOT certainty.
#
# OI ↑ + Premium ↑ = BUYING indication
# OI ↑ + Premium ↓ = WRITING indication
# OI ↓ + Premium ↑ = SHORT COVERING indication
# OI ↓ + Premium ↓ = LONG UNWINDING indication
# ============================================================

def classify_option_activity(
    option_type,
    current_ltp,
    previous_ltp,
    current_oi,
    previous_oi,
    change_oi,
    change_oi_percent,
    volume
):

    premium_change = calculate_premium_change(
        current_ltp,
        previous_ltp
    )

    oi_ratio = calculate_oi_change_ratio(
        current_oi,
        previous_oi,
        change_oi_percent
    )

    # --------------------------------------------------------
    # First snapshot
    # --------------------------------------------------------

    if (
        premium_change is None
        or
        oi_ratio is None
    ):

        return {

            "activity": "INSUFFICIENT_DATA",

            "direction": "NEUTRAL",

            "premium_change": premium_change,

            "oi_change_ratio": oi_ratio,

            "confidence": 0,

            "reason":
                "Waiting for previous premium/OI snapshot."
        }

    premium_up = (
        premium_change >=
        PREMIUM_CHANGE_THRESHOLD
    )

    premium_down = (
        premium_change <=
        -PREMIUM_CHANGE_THRESHOLD
    )

    oi_up = (
        oi_ratio >=
        OI_CHANGE_THRESHOLD
    )

    oi_down = (
        oi_ratio <=
        -OI_CHANGE_THRESHOLD
    )

    # --------------------------------------------------------
    # CE
    # --------------------------------------------------------

    if option_type == "CE":

        if oi_up and premium_up:

            activity = "CALL_BUYING"
            direction = "BULLISH"

        elif oi_up and premium_down:

            activity = "CALL_WRITING"
            direction = "BEARISH"

        elif oi_down and premium_up:

            activity = "CALL_SHORT_COVERING"
            direction = "BULLISH"

        elif oi_down and premium_down:

            activity = "CALL_LONG_UNWINDING"
            direction = "BEARISH"

        else:

            activity = "CALL_NEUTRAL"
            direction = "NEUTRAL"

    # --------------------------------------------------------
    # PE
    # --------------------------------------------------------

    else:

        if oi_up and premium_up:

            activity = "PUT_BUYING"
            direction = "BEARISH"

        elif oi_up and premium_down:

            activity = "PUT_WRITING"
            direction = "BULLISH"

        elif oi_down and premium_up:

            activity = "PUT_SHORT_COVERING"
            direction = "BEARISH"

        elif oi_down and premium_down:

            activity = "PUT_LONG_UNWINDING"
            direction = "BULLISH"

        else:

            activity = "PUT_NEUTRAL"
            direction = "NEUTRAL"

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    oi_strength = min(
        abs(oi_ratio) / 0.10,
        1.0
    )

    premium_strength = min(
        abs(premium_change) / 0.05,
        1.0
    )

    confidence = int(
        round(
            (
                oi_strength * 0.55 +
                premium_strength * 0.45
            ) * 100
        )
    )

    if volume is not None and volume > 0:

        confidence = min(
            100,
            confidence + 5
        )

    return {

        "activity": activity,

        "direction": direction,

        "premium_change":
            premium_change,

        "oi_change_ratio":
            oi_ratio,

        "confidence":
            confidence,

        "reason":
            f"{activity}: "
            f"OI change {oi_ratio * 100:.2f}% "
            f"and premium change "
            f"{premium_change * 100:.2f}%."
    }


# ============================================================
# BUILD STRIKE-WISE CHAIN
# ============================================================

def build_strike_chain(
    normalized: dict
):

    calls = normalized.get(
        "calls",
        []
    )

    puts = normalized.get(
        "puts",
        []
    )

    call_map = {

        float(row["strike"]): row

        for row in calls

        if row.get(
            "strike"
        ) is not None
    }

    put_map = {

        float(row["strike"]): row

        for row in puts

        if row.get(
            "strike"
        ) is not None
    }

    strikes = sorted(
        set(call_map.keys()) |
        set(put_map.keys())
    )

    result = []

    expiry = normalized.get(
        "expiry"
    )

    for strike in strikes:

        ce = call_map.get(
            strike
        )

        pe = put_map.get(
            strike
        )

        # ----------------------------------------------------
        # CE INTELLIGENCE
        # ----------------------------------------------------

        ce_intelligence = None

        if ce:

            previous = (
                get_previous_option_snapshot(
                    expiry,
                    "CE",
                    strike
                )
            )

            ce_intelligence = classify_option_activity(

                option_type="CE",

                current_ltp=ce.get(
                    "ltp"
                ),

                previous_ltp=(
                    previous.get("ltp")
                    if previous
                    else None
                ),

                current_oi=ce.get(
                    "oi"
                ),

                previous_oi=(
                    previous.get("oi")
                    if previous
                    else None
                ),

                change_oi=ce.get(
                    "change_oi"
                ),

                change_oi_percent=ce.get(
                    "change_oi_percent"
                ),

                volume=ce.get(
                    "volume"
                )
            )

        # ----------------------------------------------------
        # PE INTELLIGENCE
        # ----------------------------------------------------

        pe_intelligence = None

        if pe:

            previous = (
                get_previous_option_snapshot(
                    expiry,
                    "PE",
                    strike
                )
            )

            pe_intelligence = classify_option_activity(

                option_type="PE",

                current_ltp=pe.get(
                    "ltp"
                ),

                previous_ltp=(
                    previous.get("ltp")
                    if previous
                    else None
                ),

                current_oi=pe.get(
                    "oi"
                ),

                previous_oi=(
                    previous.get("oi")
                    if previous
                    else None
                ),

                change_oi=pe.get(
                    "change_oi"
                ),

                change_oi_percent=pe.get(
                    "change_oi_percent"
                ),

                volume=pe.get(
                    "volume"
                )
            )

        result.append({

            "strike": safe_float(
                strike
            ),

            "ce": ce,

            "pe": pe,

            "ce_oi": (
                ce.get("oi")
                if ce else None
            ),

            "ce_change_oi": (
                ce.get("change_oi")
                if ce else None
            ),

            "ce_ltp": (
                ce.get("ltp")
                if ce else None
            ),

            "ce_volume": (
                ce.get("volume")
                if ce else None
            ),

            "ce_activity": (
                ce_intelligence.get(
                    "activity"
                )
                if ce_intelligence
                else None
            ),

            "ce_activity_direction": (
                ce_intelligence.get(
                    "direction"
                )
                if ce_intelligence
                else None
            ),

            "ce_premium_change": (
                ce_intelligence.get(
                    "premium_change"
                )
                if ce_intelligence
                else None
            ),

            "ce_oi_change_ratio": (
                ce_intelligence.get(
                    "oi_change_ratio"
                )
                if ce_intelligence
                else None
            ),

            "pe_oi": (
                pe.get("oi")
                if pe else None
            ),

            "pe_change_oi": (
                pe.get("change_oi")
                if pe else None
            ),

            "pe_ltp": (
                pe.get("ltp")
                if pe else None
            ),

            "pe_volume": (
                pe.get("volume")
                if pe else None
            ),

            "pe_activity": (
                pe_intelligence.get(
                    "activity"
                )
                if pe_intelligence
                else None
            ),

            "pe_activity_direction": (
                pe_intelligence.get(
                    "direction"
                )
                if pe_intelligence
                else None
            ),

            "pe_premium_change": (
                pe_intelligence.get(
                    "premium_change"
                )
                if pe_intelligence
                else None
            ),

            "pe_oi_change_ratio": (
                pe_intelligence.get(
                    "oi_change_ratio"
                )
                if pe_intelligence
                else None
            )
        })

    return result


# ============================================================
# TOP OI
# ============================================================

def top_oi_rows(
    rows,
    limit=5
):

    valid = [

        row

        for row in rows

        if row.get(
            "oi"
        ) is not None

        and row.get(
            "oi"
        ) > 0
    ]

    valid.sort(
        key=lambda x: x["oi"],
        reverse=True
    )

    return valid[:limit]


# ============================================================
# TOTAL OI
# ============================================================

def total_oi(rows):

    values = [

        row.get(
            "oi"
        )

        for row in rows

        if row.get(
            "oi"
        ) is not None
    ]

    if not values:

        return 0

    return safe_int(
        sum(values)
    )


# ============================================================
# TOTAL CHANGE OI
# ============================================================

def total_change_oi(rows):

    values = [

        row.get(
            "change_oi"
        )

        for row in rows

        if row.get(
            "change_oi"
        ) is not None
    ]

    if not values:

        return 0

    return safe_float(
        sum(values),
        2
    )


# ============================================================
# PCR
# ============================================================

def calculate_pcr(
    put_oi,
    call_oi
):

    if (
        put_oi is None
        or
        call_oi is None
        or
        call_oi <= 0
    ):

        return None

    return safe_float(
        put_oi / call_oi,
        3
    )


# ============================================================
# CHANGE OI PCR
# FIXED
#
# Negative/zero denominator no longer creates nonsense PCR.
# ============================================================

def calculate_change_oi_pcr(
    put_change_oi,
    call_change_oi
):

    if (
        put_change_oi is None
        or
        call_change_oi is None
    ):

        return None

    if call_change_oi <= 0:

        return None

    return safe_float(
        put_change_oi /
        call_change_oi,
        3
    )


# ============================================================
# MAX PAIN
# ============================================================

def calculate_max_pain(
    chain_rows
):

    if not chain_rows:

        return None

    strikes = [

        row["strike"]

        for row in chain_rows

        if row.get(
            "strike"
        ) is not None
    ]

    if not strikes:

        return None

    best_strike = None
    lowest_loss = None

    for test_strike in strikes:

        total_loss = 0.0

        for row in chain_rows:

            strike = row["strike"]

            ce_oi = row.get(
                "ce_oi"
            ) or 0

            pe_oi = row.get(
                "pe_oi"
            ) or 0

            if test_strike > strike:

                total_loss += (
                    test_strike -
                    strike
                ) * ce_oi

            elif test_strike < strike:

                total_loss += (
                    strike -
                    test_strike
                ) * pe_oi

        if (
            lowest_loss is None
            or
            total_loss < lowest_loss
        ):

            lowest_loss = total_loss

            best_strike = test_strike

    return safe_float(
        best_strike
    )


# ============================================================
# OI SUPPORT / RESISTANCE
#
# Resistance:
#   Highest CE OI above/near spot
#
# Support:
#   Highest PE OI below/near spot
# ============================================================

def calculate_oi_levels(
    calls,
    puts,
    spot=None
):

    valid_calls = [
        row
        for row in calls
        if (
            row.get("oi") is not None
            and
            row.get("oi") > 0
        )
    ]

    valid_puts = [
        row
        for row in puts
        if (
            row.get("oi") is not None
            and
            row.get("oi") > 0
        )
    ]

    # --------------------------------------------------------
    # If spot unavailable use largest OI
    # --------------------------------------------------------

    if spot is None:

        top_calls = top_oi_rows(
            valid_calls,
            5
        )

        top_puts = top_oi_rows(
            valid_puts,
            5
        )

        return {

            "oi_resistance":
                (
                    top_calls[0]["strike"]
                    if top_calls
                    else None
                ),

            "oi_support":
                (
                    top_puts[0]["strike"]
                    if top_puts
                    else None
                ),

            "top_call_oi": top_calls,

            "top_put_oi": top_puts
        }

    # --------------------------------------------------------
    # CE resistance candidates
    # --------------------------------------------------------

    call_candidates = [
        row
        for row in valid_calls
        if row["strike"] >= spot
    ]

    call_candidates.sort(
        key=lambda x: x["oi"],
        reverse=True
    )

    # --------------------------------------------------------
    # PE support candidates
    # --------------------------------------------------------

    put_candidates = [
        row
        for row in valid_puts
        if row["strike"] <= spot
    ]

    put_candidates.sort(
        key=lambda x: x["oi"],
        reverse=True
    )

    # --------------------------------------------------------
    # Fallback if one side has no strike
    # --------------------------------------------------------

    if not call_candidates:

        call_candidates = sorted(
            valid_calls,
            key=lambda x: x["oi"],
            reverse=True
        )

    if not put_candidates:

        put_candidates = sorted(
            valid_puts,
            key=lambda x: x["oi"],
            reverse=True
        )

    return {

        "oi_resistance": (
            call_candidates[0]["strike"]
            if call_candidates
            else None
        ),

        "oi_support": (
            put_candidates[0]["strike"]
            if put_candidates
            else None
        ),

        "top_call_oi":
            top_oi_rows(
                valid_calls,
                5
            ),

        "top_put_oi":
            top_oi_rows(
                valid_puts,
                5
            )
    }


# ============================================================
# OI AGGREGATE INTELLIGENCE
# ============================================================

def calculate_oi_intelligence(
    chain,
    calls,
    puts,
    spot=None
):

    call_buying = 0
    call_writing = 0

    put_buying = 0
    put_writing = 0

    call_short_covering = 0
    put_short_covering = 0

    call_unwinding = 0
    put_unwinding = 0

    bullish_points = 0
    bearish_points = 0

    reasons = []

    # --------------------------------------------------------
    # Strike-level intelligence
    # --------------------------------------------------------

    for row in chain:

        ce_activity = row.get(
            "ce_activity"
        )

        pe_activity = row.get(
            "pe_activity"
        )

        # ----------------------------------------------------
        # CE
        # ----------------------------------------------------

        if ce_activity == "CALL_BUYING":

            call_buying += 1
            bullish_points += 1

        elif ce_activity == "CALL_WRITING":

            call_writing += 1
            bearish_points += 1

        elif ce_activity == "CALL_SHORT_COVERING":

            call_short_covering += 1
            bullish_points += 1

        elif ce_activity == "CALL_LONG_UNWINDING":

            call_unwinding += 1
            bearish_points += 1

        # ----------------------------------------------------
        # PE
        # ----------------------------------------------------

        if pe_activity == "PUT_BUYING":

            put_buying += 1
            bearish_points += 1

        elif pe_activity == "PUT_WRITING":

            put_writing += 1
            bullish_points += 1

        elif pe_activity == "PUT_SHORT_COVERING":

            put_short_covering += 1
            bearish_points += 1

        elif pe_activity == "PUT_LONG_UNWINDING":

            put_unwinding += 1
            bullish_points += 1

    # --------------------------------------------------------
    # Weight by proximity to spot
    #
    # Near-ATM option activity gets greater influence.
    # --------------------------------------------------------

    weighted_bullish = 0.0
    weighted_bearish = 0.0

    if spot is not None:

        for row in chain:

            strike = row.get(
                "strike"
            )

            if strike is None:
                continue

            distance = abs(
                strike - spot
            )

            distance_ratio = (
                distance /
                max(
                    spot,
                    1.0
                )
            )

            weight = max(
                0.25,
                1.0 -
                distance_ratio * 20.0
            )

            ce_direction = row.get(
                "ce_activity_direction"
            )

            pe_direction = row.get(
                "pe_activity_direction"
            )

            if ce_direction == "BULLISH":

                weighted_bullish += weight

            elif ce_direction == "BEARISH":

                weighted_bearish += weight

            if pe_direction == "BULLISH":

                weighted_bullish += weight

            elif pe_direction == "BEARISH":

                weighted_bearish += weight

    else:

        weighted_bullish = bullish_points
        weighted_bearish = bearish_points

    # --------------------------------------------------------
    # Direct four-way dominance
    # --------------------------------------------------------

    bullish_structure = (
        put_writing +
        call_buying +
        call_short_covering
    )

    bearish_structure = (
        call_writing +
        put_buying +
        put_short_covering
    )

    # --------------------------------------------------------
    # Score
    # --------------------------------------------------------

    raw_score = (
        weighted_bullish -
        weighted_bearish
    )

    structure_score = (
        bullish_structure -
        bearish_structure
    )

    # Blend
    score = (
        raw_score * 0.60 +
        structure_score * 0.40
    )

    # Normalize to -10/+10
    score = clamp(
        score,
        OI_SCORE_MIN,
        OI_SCORE_MAX
    )

    score = safe_float(
        score,
        2
    )

    # --------------------------------------------------------
    # Bias
    # --------------------------------------------------------

    if score >= 3.0:

        bias = "BULLISH"

    elif score <= -3.0:

        bias = "BEARISH"

    else:

        bias = "NEUTRAL"

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    total_activity = (
        call_buying +
        call_writing +
        put_buying +
        put_writing +
        call_short_covering +
        put_short_covering +
        call_unwinding +
        put_unwinding
    )

    if total_activity > 0:

        directional_strength = (
            abs(
                bullish_structure -
                bearish_structure
            )
            /
            total_activity
        )

    else:

        directional_strength = 0.0

    confidence = int(
        round(
            clamp(
                (
                    abs(score) / 10.0
                    * 0.60
                )
                +
                (
                    directional_strength
                    * 0.40
                ),
                0.0,
                1.0
            )
            * 100
        )
    )

    # --------------------------------------------------------
    # Reasons
    # --------------------------------------------------------

    if call_buying > 0:

        reasons.append(
            f"{call_buying} strike(s) show "
            "probable Call Buying."
        )

    if call_writing > 0:

        reasons.append(
            f"{call_writing} strike(s) show "
            "probable Call Writing."
        )

    if put_buying > 0:

        reasons.append(
            f"{put_buying} strike(s) show "
            "probable Put Buying."
        )

    if put_writing > 0:

        reasons.append(
            f"{put_writing} strike(s) show "
            "probable Put Writing."
        )

    if call_short_covering > 0:

        reasons.append(
            f"{call_short_covering} strike(s) show "
            "Call Short Covering."
        )

    if put_short_covering > 0:

        reasons.append(
            f"{put_short_covering} strike(s) show "
            "Put Short Covering."
        )

    if not reasons:

        reasons.append(
            "No strong four-way OI activity confirmed yet."
        )

    # --------------------------------------------------------
    # OI SHIFT
    # --------------------------------------------------------

    if bullish_structure >= (
        bearish_structure + 2
    ):

        oi_shift = "PUT_SUPPORT_BUILDING"

    elif bearish_structure >= (
        bullish_structure + 2
    ):

        oi_shift = "CALL_RESISTANCE_BUILDING"

    elif (
        put_writing > call_writing
        and
        put_writing > put_buying
    ):

        oi_shift = "BULLISH_PUT_WRITING"

    elif (
        call_writing > put_writing
        and
        call_writing > call_buying
    ):

        oi_shift = "BEARISH_CALL_WRITING"

    else:

        oi_shift = "NO_CLEAR_SHIFT"

    return {

        "call_buying":
            call_buying,

        "call_writing":
            call_writing,

        "put_buying":
            put_buying,

        "put_writing":
            put_writing,

        "call_short_covering":
            call_short_covering,

        "put_short_covering":
            put_short_covering,

        "call_long_unwinding":
            call_unwinding,

        "put_long_unwinding":
            put_unwinding,

        "oi_shift":
            oi_shift,

        "oi_score":
            score,

        "oi_confidence":
            confidence,

        "oi_bias":
            bias,

        "oi_reasons":
            reasons
    }


# ============================================================
# OI BIAS
# ============================================================

def calculate_oi_bias(
    pcr,
    change_pcr,
    call_change_oi,
    put_change_oi
):

    score = 0

    reasons = []

    # --------------------------------------------------------
    # PCR
    # --------------------------------------------------------

    if pcr is not None:

        if pcr >= 1.20:

            score += 2

            reasons.append(
                "PCR is bullish."
            )

        elif pcr <= 0.80:

            score -= 2

            reasons.append(
                "PCR is bearish."
            )

        elif pcr > 1.0:

            score += 1

            reasons.append(
                "PCR is mildly bullish."
            )

        elif pcr < 1.0:

            score -= 1

            reasons.append(
                "PCR is mildly bearish."
            )

    # --------------------------------------------------------
    # CHANGE OI
    # --------------------------------------------------------

    if (
        call_change_oi is not None
        and
        put_change_oi is not None
    ):

        if (
            put_change_oi > 0
            and
            call_change_oi < 0
        ):

            score += 2

            reasons.append(
                "Put OI rising while Call OI is falling."
            )

        elif (
            call_change_oi > 0
            and
            put_change_oi < 0
        ):

            score -= 2

            reasons.append(
                "Call OI rising while Put OI is falling."
            )

        elif (
            put_change_oi > 0
            and
            call_change_oi > 0
        ):

            reasons.append(
                "Both Call and Put OI are rising."
            )

    # --------------------------------------------------------
    # CHANGE PCR
    # --------------------------------------------------------

    if change_pcr is not None:

        if change_pcr > 1.10:

            score += 1

        elif change_pcr < 0.90:

            score -= 1

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    if score >= 3:

        bias = "BULLISH"

    elif score <= -3:

        bias = "BEARISH"

    else:

        bias = "NEUTRAL"

    return {

        "bias": bias,

        "score": score,

        "reasons": reasons
    }


# ============================================================
# UPDATE PREVIOUS SNAPSHOT
# ============================================================

def update_option_snapshots(
    calls,
    puts,
    expiry
):

    for row in calls:

        save_option_snapshot(

            expiry=expiry,

            option_type="CE",

            strike=row.get(
                "strike"
            ),

            ltp=row.get(
                "ltp"
            ),

            oi=row.get(
                "oi"
            ),

            volume=row.get(
                "volume"
            )
        )

    for row in puts:

        save_option_snapshot(

            expiry=expiry,

            option_type="PE",

            strike=row.get(
                "strike"
            ),

            ltp=row.get(
                "ltp"
            ),

            oi=row.get(
                "oi"
            ),

            volume=row.get(
                "volume"
            )
        )


# ============================================================
# EMPTY RESPONSE
# ============================================================

def empty_oi_response(
    expiry,
    status="NO_DATA",
    message=None
):

    if message is None:

        message = (
            "Kotak Neo returned an empty option chain."
        )

    return {

        "status": status,

        "source": "Kotak Neo",

        "underlying": KOTAK_UNDERLYING,

        "exchange": KOTAK_EXCHANGE,

        "expiry": expiry,

        "total_call_oi": 0,

        "total_put_oi": 0,

        "call_change_oi": 0,

        "put_change_oi": 0,

        "pcr": None,

        "change_oi_pcr": None,

        "max_pain": None,

        "oi_support": None,

        "oi_resistance": None,

        "oi_bias": "UNKNOWN",

        "oi_score": 0,

        "oi_confidence": 0,

        "call_buying": 0,

        "call_writing": 0,

        "put_buying": 0,

        "put_writing": 0,

        "oi_shift": "UNKNOWN",

        "oi_reasons": [
            message
        ],

        "top_call_oi": [],

        "top_put_oi": [],

        "chain": [],

        "timestamp":
            datetime.now().isoformat()
    }


# ============================================================
# COMPLETE OI ANALYSIS
# ============================================================

def get_oi_analysis(
    expiry: Optional[str] = None,
    count: int = DEFAULT_COUNT
):

    requested_expiry = expiry

    try:

        if expiry is None:

            expiry = get_nearest_expiry()

        raw = fetch_option_chain(
            expiry=expiry,
            count=count
        )

        if not isinstance(raw, dict):

            return empty_oi_response(
                expiry,
                "ERROR",
                "Invalid Kotak option-chain response."
            )

        normalized = normalize_chain(
            raw,
            expiry
        )

        calls = normalized[
            "calls"
        ]

        puts = normalized[
            "puts"
        ]

        # ----------------------------------------------------
        # NO DATA
        # ----------------------------------------------------

        if (
            len(calls) == 0
            and
            len(puts) == 0
        ):

            response = empty_oi_response(
                expiry
            )

            response[
                "requested_expiry"
            ] = raw.get(
                "_requested_expiry",
                requested_expiry
            )

            response[
                "actual_expiry"
            ] = raw.get(
                "_actual_expiry",
                expiry
            )

            response[
                "chain_attempt"
            ] = raw.get(
                "_chain_attempt",
                "no_data"
            )

            response[
                "api_stat"
            ] = raw.get(
                "stat"
            )

            response[
                "api_code"
            ] = raw.get(
                "stCode"
            )

            response[
                "api_message"
            ] = (
                raw.get(
                    "errMsg"
                )
                or
                raw.get(
                    "desc"
                )
            )

            return response

        # ----------------------------------------------------
        # SPOT
        # ----------------------------------------------------

        spot_ltp = (
            normalized
            .get("spot", {})
            .get("ltp")
        )

        # ----------------------------------------------------
        # CHAIN
        # ----------------------------------------------------

        chain = build_strike_chain(
            normalized
        )

        # ----------------------------------------------------
        # TOTALS
        # ----------------------------------------------------

        total_call_oi = total_oi(
            calls
        )

        total_put_oi = total_oi(
            puts
        )

        call_change_oi = total_change_oi(
            calls
        )

        put_change_oi = total_change_oi(
            puts
        )

        # ----------------------------------------------------
        # PCR
        # ----------------------------------------------------

        pcr = calculate_pcr(
            total_put_oi,
            total_call_oi
        )

        change_pcr = calculate_change_oi_pcr(
            put_change_oi,
            call_change_oi
        )

        # ----------------------------------------------------
        # MAX PAIN
        # ----------------------------------------------------

        max_pain = calculate_max_pain(
            chain
        )

        # ----------------------------------------------------
        # LEVELS
        # ----------------------------------------------------

        levels = calculate_oi_levels(
            calls,
            puts,
            spot_ltp
        )

        # ----------------------------------------------------
        # BASIC OI BIAS
        # ----------------------------------------------------

        basic_bias = calculate_oi_bias(

            pcr=pcr,

            change_pcr=change_pcr,

            call_change_oi=call_change_oi,

            put_change_oi=put_change_oi
        )

        # ----------------------------------------------------
        # ADVANCED OI INTELLIGENCE
        # ----------------------------------------------------

        intelligence = calculate_oi_intelligence(

            chain=chain,

            calls=calls,

            puts=puts,

            spot=spot_ltp
        )

        # ----------------------------------------------------
        # COMBINED SCORE
        #
        # Advanced four-way intelligence gets 70%.
        # PCR/OI aggregate gets 30%.
        # ----------------------------------------------------

        advanced_score = (
            intelligence.get(
                "oi_score"
            )
            or 0
        )

        basic_score = (
            basic_bias.get(
                "score"
            )
            or 0
        )

        combined_score = (
            advanced_score * 0.70
            +
            basic_score * 0.30
        )

        combined_score = safe_float(
            clamp(
                combined_score,
                OI_SCORE_MIN,
                OI_SCORE_MAX
            ),
            2
        )

        # ----------------------------------------------------
        # FINAL OI BIAS
        # ----------------------------------------------------

        if combined_score >= 3.0:

            final_bias = "BULLISH"

        elif combined_score <= -3.0:

            final_bias = "BEARISH"

        else:

            final_bias = "NEUTRAL"

        # ----------------------------------------------------
        # FINAL CONFIDENCE
        # ----------------------------------------------------

        advanced_confidence = (
            intelligence.get(
                "oi_confidence"
            )
            or 0
        )

        confidence = int(
            round(
                advanced_confidence * 0.70
                +
                min(
                    abs(
                        basic_score
                    ) / 5.0,
                    1.0
                ) * 100 * 0.30
            )
        )

        confidence = int(
            clamp(
                confidence,
                0,
                100
            )
        )

        # ----------------------------------------------------
        # COMBINED REASONS
        # ----------------------------------------------------

        combined_reasons = []

        combined_reasons.extend(
            intelligence.get(
                "oi_reasons",
                []
            )
        )

        combined_reasons.extend(
            basic_bias.get(
                "reasons",
                []
            )
        )

        # Remove duplicates
        combined_reasons = list(
            dict.fromkeys(
                combined_reasons
            )
        )

        # ----------------------------------------------------
        # UPDATE SNAPSHOT
        #
        # IMPORTANT:
        # Do this AFTER intelligence calculation.
        # Otherwise current LTP becomes previous LTP.
        # ----------------------------------------------------

        update_option_snapshots(
            calls=calls,
            puts=puts,
            expiry=expiry
        )

        # ----------------------------------------------------
        # SUCCESS RESPONSE
        # ----------------------------------------------------

        return {

            "status": "OK",

            "source": "Kotak Neo",

            "underlying":
                KOTAK_UNDERLYING,

            "exchange":
                KOTAK_EXCHANGE,

            "expiry":
                expiry,

            "requested_expiry":
                raw.get(
                    "_requested_expiry",
                    requested_expiry
                ),

            "actual_expiry":
                raw.get(
                    "_actual_expiry",
                    expiry
                ),

            "chain_attempt":
                raw.get(
                    "_chain_attempt",
                    "explicit_expiry"
                ),

            "lot_size":
                normalized[
                    "lot_size"
                ],

            "multiplier":
                normalized[
                    "multiplier"
                ],

            "spot":
                normalized[
                    "spot"
                ],

            "future":
                normalized[
                    "future"
                ],

            "total_call_oi":
                total_call_oi,

            "total_put_oi":
                total_put_oi,

            "call_change_oi":
                call_change_oi,

            "put_change_oi":
                put_change_oi,

            "pcr":
                pcr,

            "change_oi_pcr":
                change_pcr,

            "max_pain":
                max_pain,

            "oi_support":
                levels[
                    "oi_support"
                ],

            "oi_resistance":
                levels[
                    "oi_resistance"
                ],

            # ------------------------------------------------
            # FINAL OI INTELLIGENCE
            # ------------------------------------------------

            "oi_bias":
                final_bias,

            "oi_score":
                combined_score,

            "oi_confidence":
                confidence,

            "oi_shift":
                intelligence.get(
                    "oi_shift"
                ),

            # ------------------------------------------------
            # FOUR-WAY
            # ------------------------------------------------

            "call_buying":
                intelligence.get(
                    "call_buying"
                ),

            "call_writing":
                intelligence.get(
                    "call_writing"
                ),

            "put_buying":
                intelligence.get(
                    "put_buying"
                ),

            "put_writing":
                intelligence.get(
                    "put_writing"
                ),

            "call_short_covering":
                intelligence.get(
                    "call_short_covering"
                ),

            "put_short_covering":
                intelligence.get(
                    "put_short_covering"
                ),

            "call_long_unwinding":
                intelligence.get(
                    "call_long_unwinding"
                ),

            "put_long_unwinding":
                intelligence.get(
                    "put_long_unwinding"
                ),

            "oi_reasons":
                combined_reasons,

            "top_call_oi":
                levels[
                    "top_call_oi"
                ],

            "top_put_oi":
                levels[
                    "top_put_oi"
                ],

            "chain":
                chain,

            "timestamp":
                datetime.now().isoformat()
        }

    except Exception as exc:

        return {

            "status": "ERROR",

            "source": "Kotak Neo",

            "underlying":
                KOTAK_UNDERLYING,

            "exchange":
                KOTAK_EXCHANGE,

            "expiry":
                expiry,

            "requested_expiry":
                requested_expiry,

            "message":
                str(exc),

            "error_type":
                str(type(exc)),

            "total_call_oi": 0,

            "total_put_oi": 0,

            "call_change_oi": 0,

            "put_change_oi": 0,

            "pcr": None,

            "change_oi_pcr": None,

            "max_pain": None,

            "oi_support": None,

            "oi_resistance": None,

            "oi_bias": "UNKNOWN",

            "oi_score": 0,

            "oi_confidence": 0,

            "call_buying": 0,

            "call_writing": 0,

            "put_buying": 0,

            "put_writing": 0,

            "oi_shift": "UNKNOWN",

            "oi_reasons": [
                "Kotak OI analysis failed."
            ],

            "top_call_oi": [],

            "top_put_oi": [],

            "chain": [],

            "timestamp":
                datetime.now().isoformat()
        }


# ============================================================
# KOTAK STATUS
# ============================================================

def kotak_status():

    consumer_key = get_env(
        "NEO_CONSUMER_KEY"
    )

    return {

        "provider":
            "Kotak Neo",

        "configured":
            bool(consumer_key),

        "market_data":
            True,

        "option_chain":
            True,

        "oi_intelligence":
            True,

        "four_way_analysis":
            True,

        "order_placement":
            False
            }
