# ============================================================
# NIFTY AI TRADER — KOTAK NEO OI ENGINE
# Version: OI-1.1
#
# Market Data Only
# NO ORDER PLACEMENT
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

_oi_cache = {}
_oi_cache_lock = threading.Lock()


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
# CHECK OPTION CHAIN DATA
# ============================================================

def has_option_chain_data(response):

    if not isinstance(response, dict):

        return False

    data = response.get(
        "data",
        {}
    )

    if not isinstance(data, dict):

        return False

    calls = data.get(
        "call",
        []
    )

    puts = data.get(
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

    # --------------------------------------------------------
    # ATTEMPT 1
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # IF DATA EXISTS, RETURN IT
    # --------------------------------------------------------

    if has_option_chain_data(response):

        response["_requested_expiry"] = (
            requested_expiry
        )

        response["_actual_expiry"] = (
            expiry
        )

        response["_chain_attempt"] = (
            "explicit_expiry"
        )

        set_cached_oi(
            cache_key,
            response
        )

        return response

    # --------------------------------------------------------
    # ATTEMPT 2
    #
    # Ask Kotak for nearest expiry automatically.
    # This protects against an expiry-specific empty response.
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
            ] = "nearest_expiry_fallback"

            set_cached_oi(
                cache_key,
                fallback_response
            )

            return fallback_response

    except Exception:
        pass

    # --------------------------------------------------------
    # NO DATA
    # --------------------------------------------------------

    response["_requested_expiry"] = (
        requested_expiry
    )

    response["_actual_expiry"] = (
        expiry
    )

    response["_chain_attempt"] = (
        "no_data"
    )

    return response


# ============================================================
# OPTION ROW NORMALIZER
# ============================================================

def normalize_option(
    item,
    option_type: str
):

    if not isinstance(item, dict):

        return None

    instrument = item.get(
        "instrument",
        {}
    )

    quote = item.get(
        "quote",
        {}
    )

    oi = item.get(
        "openInterest",
        {}
    )

    if not isinstance(instrument, dict):
        instrument = {}

    if not isinstance(quote, dict):
        quote = {}

    if not isinstance(oi, dict):
        oi = {}

    strike = safe_float(
        instrument.get(
            "strikePrice"
        )
    )

    if strike is None:
        return None

    current_oi = safe_int(
        oi.get(
            "current"
        )
    )

    previous_oi = safe_int(
        oi.get(
            "previous"
        )
    )

    change_oi = safe_int(
        oi.get(
            "change"
        )
    )

    change_pct = safe_float(
        oi.get(
            "changePct"
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
            "volume"
        )
    )

    return {

        "option_type": option_type,

        "strike": strike,

        "symbol": instrument.get(
            "symbol"
        ),

        "neo_symbol": instrument.get(
            "neoSymbol"
        ),

        "moneyness": instrument.get(
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

    data = raw_response.get(
        "data",
        {}
    )

    if not isinstance(data, dict):

        raise RuntimeError(
            "Kotak option-chain data is missing."
        )

    common = data.get(
        "common_data",
        {}
    )

    if not isinstance(common, dict):
        common = {}

    calls_raw = data.get(
        "call",
        []
    )

    puts_raw = data.get(
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

    return {

        "expiry": expiry,

        "underlying": KOTAK_UNDERLYING,

        "exchange": KOTAK_EXCHANGE,

        "lot_size": safe_int(
            common.get(
                "mktLot"
            )
        ),

        "calls": calls,

        "puts": puts
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
        if row.get("strike") is not None
    }

    put_map = {
        float(row["strike"]): row
        for row in puts
        if row.get("strike") is not None
    }

    strikes = sorted(
        set(call_map.keys()) |
        set(put_map.keys())
    )

    result = []

    for strike in strikes:

        ce = call_map.get(
            strike
        )

        pe = put_map.get(
            strike
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

        if row.get("oi") is not None
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

        row.get("oi")

        for row in rows

        if row.get("oi") is not None
    ]

    if not values:
        return 0

    return safe_int(
        sum(values)
    )


def total_change_oi(rows):

    values = [

        row.get("change_oi")

        for row in rows

        if row.get("change_oi") is not None
    ]

    if not values:
        return 0

    return safe_int(
        sum(values)
    )


# ============================================================
# PCR
# ============================================================

def calculate_pcr(
    put_oi,
    call_oi
):

    if (
        put_oi is None or
        call_oi is None or
        call_oi <= 0
    ):

        return None

    return safe_float(
        put_oi / call_oi,
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

        if row.get("strike") is not None
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
# ============================================================

def calculate_oi_levels(
    calls,
    puts
):

    top_calls = top_oi_rows(
        calls,
        5
    )

    top_puts = top_oi_rows(
        puts,
        5
    )

    oi_resistance = None

    oi_support = None

    if top_calls:

        oi_resistance = (
            top_calls[0].get(
                "strike"
            )
        )

    if top_puts:

        oi_support = (
            top_puts[0].get(
                "strike"
            )
        )

    return {

        "oi_resistance": safe_float(
            oi_resistance
        ),

        "oi_support": safe_float(
            oi_support
        ),

        "top_call_oi": top_calls,

        "top_put_oi": top_puts
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
                "PCR is bullish"
            )

        elif pcr <= 0.80:

            score -= 2

            reasons.append(
                "PCR is bearish"
            )

        elif pcr > 1.0:

            score += 1

        elif pcr < 1.0:

            score -= 1

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
                "Put OI rising while Call OI is falling"
            )

        elif (
            call_change_oi > 0
            and
            put_change_oi < 0
        ):

            score -= 2

            reasons.append(
                "Call OI rising while Put OI is falling"
            )

        elif (
            put_change_oi > 0
            and
            call_change_oi > 0
        ):

            reasons.append(
                "Both Call and Put OI are rising"
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
    # FINAL BIAS
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
# COMPLETE OI ANALYSIS
# ============================================================

def get_oi_analysis(
    expiry: Optional[str] = None,
    count: int = DEFAULT_COUNT
):

    requested_expiry = expiry

    if expiry is None:

        expiry = get_nearest_expiry()

    raw = fetch_option_chain(
        expiry=expiry,
        count=count
    )

    # --------------------------------------------------------
    # VALIDATE RAW RESPONSE
    # --------------------------------------------------------

    if not isinstance(raw, dict):

        return {

            "status": "ERROR",

            "source": "Kotak Neo",

            "underlying": "NIFTY",

            "exchange": KOTAK_EXCHANGE,

            "expiry": expiry,

            "message": (
                "Invalid option-chain response."
            ),

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

            "oi_reasons": [],

            "top_call_oi": [],

            "top_put_oi": [],

            "chain": [],

            "timestamp": datetime.now().isoformat()
        }

    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # NO DATA CHECK
    # --------------------------------------------------------

    if (
        len(calls) == 0
        and
        len(puts) == 0
    ):

        data = raw.get(
            "data",
            {}
        )

        if not isinstance(data, dict):
            data = {}

        return {

            "status": "NO_DATA",

            "source": "Kotak Neo",

            "underlying": "NIFTY",

            "exchange": KOTAK_EXCHANGE,

            "expiry": expiry,

            "requested_expiry": raw.get(
                "_requested_expiry"
            ),

            "actual_expiry": raw.get(
                "_actual_expiry",
                expiry
            ),

            "chain_attempt": raw.get(
                "_chain_attempt",
                "no_data"
            ),

            "api_stat": raw.get(
                "stat"
            ),

            "api_code": raw.get(
                "stCode"
            ),

            "api_message": (
                raw.get("errMsg")
                or
                raw.get("desc")
            ),

            "common_data": data.get(
                "common_data",
                {}
            ),

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

            "oi_reasons": [

                "Kotak Neo returned an empty option chain."

            ],

            "top_call_oi": [],

            "top_put_oi": [],

            "chain": [],

            "timestamp": datetime.now().isoformat()
        }

    # --------------------------------------------------------
    # BUILD ANALYSIS
    # --------------------------------------------------------

    chain = build_strike_chain(
        normalized
    )

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

    pcr = calculate_pcr(
        total_put_oi,
        total_call_oi
    )

    change_pcr = calculate_pcr(
        put_change_oi,
        call_change_oi
    )

    max_pain = calculate_max_pain(
        chain
    )

    levels = calculate_oi_levels(
        calls,
        puts
    )

    bias = calculate_oi_bias(
        pcr=pcr,
        change_pcr=change_pcr,
        call_change_oi=call_change_oi,
        put_change_oi=put_change_oi
    )

    # --------------------------------------------------------
    # SUCCESS
    # --------------------------------------------------------

    return {

        "status": "OK",

        "source": "Kotak Neo",

        "underlying": "NIFTY",

        "exchange": KOTAK_EXCHANGE,

        "expiry": expiry,

        "requested_expiry": raw.get(
            "_requested_expiry"
        ),

        "actual_expiry": raw.get(
            "_actual_expiry",
            expiry
        ),

        "chain_attempt": raw.get(
            "_chain_attempt",
            "explicit_expiry"
        ),

        "lot_size": normalized[
            "lot_size"
        ],

        "total_call_oi": total_call_oi,

        "total_put_oi": total_put_oi,

        "call_change_oi": call_change_oi,

        "put_change_oi": put_change_oi,

        "pcr": pcr,

        "change_oi_pcr": change_pcr,

        "max_pain": max_pain,

        "oi_support": levels[
            "oi_support"
        ],

        "oi_resistance": levels[
            "oi_resistance"
        ],

        "oi_bias": bias[
            "bias"
        ],

        "oi_score": bias[
            "score"
        ],

        "oi_reasons": bias[
            "reasons"
        ],

        "top_call_oi": levels[
            "top_call_oi"
        ],

        "top_put_oi": levels[
            "top_put_oi"
        ],

        "chain": chain,

        "timestamp": datetime.now().isoformat()
    }


# ============================================================
# STATUS
# ============================================================

def kotak_status():

    consumer_key = get_env(
        "NEO_CONSUMER_KEY"
    )

    return {

        "provider": "Kotak Neo",

        "configured": bool(
            consumer_key
        ),

        "market_data": True,

        "option_chain": True,

        "order_placement": False
    }
