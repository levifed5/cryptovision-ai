from flask import Flask, render_template, request
import requests
import numpy as np
import time
import os
import sqlite3
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta
from pytrends.request import TrendReq

app = Flask(__name__)

# Your free CoinGecko API key, read from an environment variable so it
# never gets hardcoded into the source code (and never ends up on GitHub).
COINGECKO_API_KEY = os.environ.get("COINGECKO_API_KEY", "")
COINGECKO_HEADERS = {"x-cg-demo-api-key": COINGECKO_API_KEY} if COINGECKO_API_KEY else {}

# Coins the user can choose from: CoinGecko id -> display name
AVAILABLE_COINS = {
    "bitcoin": "Bitcoin",
    "ethereum": "Ethereum",
    "solana": "Solana",
    "dogecoin": "Dogecoin",
    "cardano": "Cardano",
    "ripple": "XRP",
    "litecoin": "Litecoin",
    "polkadot": "Polkadot",
    "chainlink": "Chainlink",
    "avalanche-2": "Avalanche",
    "tron": "Tron",
    "shiba-inu": "Shiba Inu",
    "polygon-ecosystem-token": "Polygon",
    "stellar": "Stellar",
}

DB_PATH = "predictions.db"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            coin_id TEXT,
            predicted_for_date TEXT,
            predicted_price REAL,
            trend_price REAL,
            momentum_price REAL,
            crossover_price REAL,
            made_at TEXT,
            PRIMARY KEY (coin_id, predicted_for_date)
        )
        """
    )
    # If the table already existed from before this feature was added,
    # add the new columns onto it (ignored if they're already there).
    for col in ("trend_price REAL", "momentum_price REAL", "crossover_price REAL"):
        try:
            conn.execute(f"ALTER TABLE predictions ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()
    conn.close()


init_db()

# Simple in-memory caches so we don't hammer free APIs on every request
_chart_cache = {}    # (coin_id, days) -> (dates, values, fetched_at)
_details_cache = {}  # coin_id -> (details_dict, fetched_at)
_news_cache = {"items": None, "fetched_at": 0}
CACHE_SECONDS = 600  # 10 minutes -- was 2, raised to cut down on rate-limit hits
NEWS_CACHE_SECONDS = 600


def get_historical_prices(coin_id, days=30):
    """Fetch daily closing prices for the past `days` days, using a
    cache to avoid repeated requests for the same data. If we get
    rate-limited, wait briefly and retry once before giving up."""
    key = (coin_id, days)
    now = time.time()
    if key in _chart_cache:
        cached_dates, cached_values, fetched_at = _chart_cache[key]
        if now - fetched_at < CACHE_SECONDS:
            return cached_dates, cached_values

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {"vs_currency": "usd", "days": days, "interval": "daily"}

    response = requests.get(url, params=params, headers=COINGECKO_HEADERS, timeout=10)
    if response.status_code == 429:
        time.sleep(2)  # rate limits on the free tier often clear within a couple seconds
        response = requests.get(url, params=params, headers=COINGECKO_HEADERS, timeout=10)

    if response.status_code == 429:
        # Still rate-limited after retrying -- show slightly stale cached
        # data if we have any, rather than an error.
        if key in _chart_cache:
            cached_dates, cached_values, _ = _chart_cache[key]
            return cached_dates, cached_values
        raise ValueError("CoinGecko is rate-limiting us right now. Wait a few seconds and try again.")
    if response.status_code != 200:
        if key in _chart_cache:
            cached_dates, cached_values, _ = _chart_cache[key]
            return cached_dates, cached_values
        raise ValueError(f"CoinGecko returned an error (status {response.status_code}). Try again shortly.")

    data = response.json()
    if "prices" not in data:
        raise ValueError("CoinGecko didn't return price data. Try again in a moment.")

    prices = data["prices"]
    dates = [datetime.utcfromtimestamp(p[0] / 1000).strftime("%b %d") for p in prices]
    values = [p[1] for p in prices]

    _chart_cache[key] = (dates, values, now)
    return dates, values


def get_coin_details(coin_id):
    """Get 24h and 7d percent change from CoinGecko's coin detail endpoint."""
    now = time.time()
    if coin_id in _details_cache:
        cached, fetched_at = _details_cache[coin_id]
        if now - fetched_at < CACHE_SECONDS:
            return cached

    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
    params = {
        "localization": "false",
        "tickers": "false",
        "market_data": "true",
        "community_data": "false",
        "developer_data": "false",
    }
    response = requests.get(url, params=params, headers=COINGECKO_HEADERS, timeout=10)
    if response.status_code == 429:
        time.sleep(2)
        response = requests.get(url, params=params, headers=COINGECKO_HEADERS, timeout=10)

    if response.status_code == 429:
        if coin_id in _details_cache:
            cached, _ = _details_cache[coin_id]
            return cached
        raise ValueError("CoinGecko is rate-limiting us right now. Wait a few seconds and try again.")
    if response.status_code != 200:
        if coin_id in _details_cache:
            cached, _ = _details_cache[coin_id]
            return cached
        raise ValueError(f"CoinGecko returned an error (status {response.status_code}). Try again shortly.")

    data = response.json()
    market_data = data.get("market_data", {})
    details = {
        "change_24h": market_data.get("price_change_percentage_24h"),
        "change_7d": market_data.get("price_change_percentage_7d"),
    }
    _details_cache[coin_id] = (details, now)
    return details


def get_news():
    """Pull recent crypto headlines from CoinDesk's public RSS feed
    (no API key required). Cached for 10 minutes.
    Returns (items, error_message). error_message is None on success."""
    now = time.time()
    if _news_cache["items"] is not None and now - _news_cache["fetched_at"] < NEWS_CACHE_SECONDS:
        return _news_cache["items"], None

    try:
        response = requests.get("https://www.coindesk.com/arc/outboundfeeds/rss/", timeout=8)
        if response.status_code != 200:
            raise ValueError(f"news feed returned status {response.status_code}")

        root = ET.fromstring(response.content)
        items = []
        for item in root.findall(".//item")[:6]:
            title = item.findtext("title")
            link = item.findtext("link")
            if title and link:
                items.append({"title": title.strip(), "link": link.strip()})

        if not items:
            raise ValueError("news feed returned no headlines")

        _news_cache["items"] = items
        _news_cache["fetched_at"] = now
        return items, None
    except Exception:
        # If we have older cached headlines, show those rather than nothing.
        if _news_cache["items"]:
            return _news_cache["items"], None
        return [], "Couldn't load the news feed right now. Try refreshing in a bit."


def get_all_current_prices():
    """Get the current USD price for every coin in AVAILABLE_COINS in a
    single request, for the portfolio tracker page."""
    cache_key = "all_prices"
    now = time.time()
    if cache_key in _details_cache:
        cached, fetched_at = _details_cache[cache_key]
        if now - fetched_at < CACHE_SECONDS:
            return cached

    ids = ",".join(AVAILABLE_COINS.keys())
    url = "https://api.coingecko.com/api/v3/simple/price"
    params = {"ids": ids, "vs_currencies": "usd"}
    response = requests.get(url, params=params, headers=COINGECKO_HEADERS, timeout=10)

    if response.status_code != 200:
        if cache_key in _details_cache:
            cached, _ = _details_cache[cache_key]
            return cached
        raise ValueError("Couldn't load current prices right now. Try again shortly.")

    data = response.json()
    prices = {coin_id: data.get(coin_id, {}).get("usd") for coin_id in AVAILABLE_COINS}
    _details_cache[cache_key] = (prices, now)
    return prices


def trend_model(values):
    """Fit a trend line through recent prices and extrapolate one day
    forward, weighting recent days more heavily than older ones."""
    x = np.arange(len(values))
    y = np.array(values)
    weights = np.linspace(1, 3, len(values))  # later days count up to 3x as much
    slope, intercept = np.polyfit(x, y, 1, w=weights)
    next_x = len(values)
    return slope * next_x + intercept


def momentum_model(values):
    """A different, simpler approach: look at the average daily percent
    change over the last 5 days and assume that pace continues for one
    more day. Reacts faster to short bursts than the trend model does."""
    recent = values[-6:] if len(values) >= 6 else values
    daily_changes = [(recent[i] - recent[i - 1]) / recent[i - 1] for i in range(1, len(recent))]
    avg_daily_change = sum(daily_changes) / len(daily_changes) if daily_changes else 0
    return values[-1] * (1 + avg_daily_change)


def moving_average_crossover_model(values):
    """A classic technical-analysis technique real traders actually use:
    compare a short-term moving average (last 5 days) to a longer-term one
    (last 15 days, or as many as we have). When the short average is above
    the long average, that's historically read as a bullish signal (and
    vice versa) -- so we nudge the current price up or down slightly based
    on how far apart the two averages are, as a rough next-day estimate."""
    short_window = min(5, len(values))
    long_window = min(15, len(values))
    short_avg = sum(values[-short_window:]) / short_window
    long_avg = sum(values[-long_window:]) / long_window

    if long_avg == 0:
        return values[-1]

    # How far the short average has pulled away from the long average,
    # as a percentage -- used as a small nudge on tomorrow's price.
    signal_strength = (short_avg - long_avg) / long_avg
    return values[-1] * (1 + signal_strength * 0.5)  # damped so it doesn't overreact


def get_model_weights(coin_id):
    """Look at how each model has actually performed for this coin
    recently (using logged predictions we can now check against actual
    prices) and weight the better-performing ones more heavily going
    forward. This is the 'learns from its own track record' part --
    it's still simple, but it is genuinely adaptive, not fixed.
    Falls back to an even three-way split if there isn't enough history yet."""
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT predicted_for_date, trend_price, momentum_price, crossover_price FROM predictions "
        "WHERE coin_id = ? AND trend_price IS NOT NULL AND momentum_price IS NOT NULL "
        "ORDER BY predicted_for_date DESC LIMIT 10",
        (coin_id,),
    ).fetchall()
    conn.close()

    if len(rows) < 3:
        return 1 / 3, 1 / 3, 1 / 3  # not enough history yet -- start even

    try:
        dates, prices = get_historical_prices(coin_id, days=30)
    except (ValueError, requests.RequestException):
        return 1 / 3, 1 / 3, 1 / 3

    trend_errors, momentum_errors, crossover_errors = [], [], []
    for predicted_for_date, trend_price, momentum_price, crossover_price in rows:
        pf_date = datetime.fromisoformat(predicted_for_date).date()
        label = pf_date.strftime("%b %d")
        if label in dates:
            actual = prices[dates.index(label)]
            if actual:
                trend_errors.append(abs(trend_price - actual) / actual)
                momentum_errors.append(abs(momentum_price - actual) / actual)
                if crossover_price is not None:
                    crossover_errors.append(abs(crossover_price - actual) / actual)

    if len(trend_errors) < 2:
        return 1 / 3, 1 / 3, 1 / 3

    avg_trend_error = sum(trend_errors) / len(trend_errors)
    avg_momentum_error = sum(momentum_errors) / len(momentum_errors)
    avg_crossover_error = (
        sum(crossover_errors) / len(crossover_errors) if crossover_errors else avg_trend_error
    )

    # Inverse-error weighting: the model with the smaller average error
    # gets the bigger weight. Add a tiny constant so we never divide by zero.
    trend_score = 1 / (avg_trend_error + 0.001)
    momentum_score = 1 / (avg_momentum_error + 0.001)
    crossover_score = 1 / (avg_crossover_error + 0.001)
    total = trend_score + momentum_score + crossover_score
    return trend_score / total, momentum_score / total, crossover_score / total


def get_search_interest(coin_name):
    """Get how much people are currently searching for this coin on Google,
    via the unofficial 'pytrends' library (Google doesn't offer a real free
    API for this). Returns a 0-100 score, or None if it fails -- this can
    happen since it's unofficial and sometimes gets rate-limited.
    Cached for an hour per coin since this is the most fragile data source."""
    cache_key = f"trends_{coin_name}"
    now = time.time()
    if cache_key in _details_cache:
        cached, fetched_at = _details_cache[cache_key]
        if now - fetched_at < 3600:  # 1 hour cache -- trends move slowly anyway
            return cached
    try:
        pytrends = TrendReq(hl="en-US", tz=360, timeout=(5, 10))
        pytrends.build_payload([coin_name], timeframe="now 7-d")
        df = pytrends.interest_over_time()
        if df.empty:
            raise ValueError("no trends data")
        score = int(df[coin_name].iloc[-1])
        _details_cache[cache_key] = (score, now)
        return score
    except Exception:
        # Fragile/unofficial source -- if it fails, fall back to any
        # previously cached value rather than breaking the page.
        if cache_key in _details_cache:
            return _details_cache[cache_key][0]
        return None


def get_fear_greed_index():
    """Fetch the Crypto Fear & Greed Index -- a real, widely-used public
    sentiment gauge (0 = extreme fear, 100 = extreme greed), from
    alternative.me's free API. No key required."""
    cache_key = "fear_greed"
    now = time.time()
    if cache_key in _details_cache:
        cached, fetched_at = _details_cache[cache_key]
        if now - fetched_at < CACHE_SECONDS:
            return cached
    try:
        response = requests.get("https://api.alternative.me/fng/", timeout=8)
        data = response.json()
        value = int(data["data"][0]["value"])
        label = data["data"][0]["value_classification"]
        result = {"value": value, "label": label}
        _details_cache[cache_key] = (result, now)
        return result
    except Exception:
        return None


POSITIVE_WORDS = {"rally", "surge", "gain", "gains", "jump", "soar", "soars", "bullish",
                   "rebound", "rise", "rises", "high", "record", "breakout", "climb"}
NEGATIVE_WORDS = {"crash", "plunge", "drop", "drops", "fall", "falls", "bearish",
                   "sell-off", "selloff", "low", "decline", "slump", "tumble", "loss"}


def score_news_sentiment(news_items):
    """A very rough keyword-based sentiment score from headline text:
    positive count minus negative count, scaled to roughly -1 to 1.
    This is a genuinely simple technique (real sentiment analysis uses
    trained language models) but it's transparent and free."""
    if not news_items:
        return 0
    pos, neg = 0, 0
    for item in news_items:
        words = set(item["title"].lower().replace(",", "").replace(".", "").split())
        pos += len(words & POSITIVE_WORDS)
        neg += len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0
    return (pos - neg) / total  # ranges from -1 (all negative) to 1 (all positive)


def calculate_volatility(values):
    y = np.array(values)
    pct_std = (np.std(y) / np.mean(y)) * 100
    if pct_std < 3:
        label = "Low"
    elif pct_std < 8:
        label = "Medium"
    else:
        label = "High"
    return label, pct_std


def record_prediction(coin_id, blended_price, trend_price, momentum_price, crossover_price):
    today = date.today()
    predicted_for = today + timedelta(days=1)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR IGNORE INTO predictions "
        "(coin_id, predicted_for_date, predicted_price, trend_price, momentum_price, crossover_price, made_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (coin_id, predicted_for.isoformat(), blended_price, trend_price, momentum_price, crossover_price, today.isoformat()),
    )
    conn.commit()
    conn.close()


def get_prediction_history(coin_id, actual_dates, actual_values):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT predicted_for_date, predicted_price FROM predictions WHERE coin_id = ? ORDER BY predicted_for_date DESC LIMIT 14",
        (coin_id,),
    ).fetchall()
    conn.close()

    history = []
    for predicted_for_date, predicted_price in rows:
        pf_date = datetime.fromisoformat(predicted_for_date).date()
        label = pf_date.strftime("%b %d")
        actual_price = None
        if label in actual_dates:
            actual_price = actual_values[actual_dates.index(label)]

        error_pct = None
        if actual_price is not None and actual_price != 0:
            error_pct = abs(predicted_price - actual_price) / actual_price * 100

        history.append(
            {
                "date": pf_date.strftime("%b %d, %Y"),
                "predicted": round(predicted_price, 2),
                "actual": round(actual_price, 2) if actual_price is not None else None,
                "error_pct": round(error_pct, 1) if error_pct is not None else None,
            }
        )
    return history


@app.route("/")
def home():
    coin_id = request.args.get("coin", "bitcoin")
    if coin_id not in AVAILABLE_COINS:
        coin_id = "bitcoin"

    compare_id = request.args.get("compare", "")
    if compare_id not in AVAILABLE_COINS:
        compare_id = ""

    days = request.args.get("days", "30")
    if days not in ("7", "30", "90"):
        days = "30"
    days = int(days)

    try:
        dates, prices = get_historical_prices(coin_id, days=days)
        details = get_coin_details(coin_id)
    except (ValueError, requests.RequestException) as e:
        return render_template(
            "error.html", message=str(e), coin_id=coin_id, available_coins=AVAILABLE_COINS
        )

    trend_price = trend_model(prices)
    momentum_price = momentum_model(prices)
    crossover_price = moving_average_crossover_model(prices)
    trend_weight, momentum_weight, crossover_weight = get_model_weights(coin_id)
    predicted = (
        trend_price * trend_weight
        + momentum_price * momentum_weight
        + crossover_price * crossover_weight
    )

    current = prices[-1]
    percent_change = (predicted - current) / current * 100
    volatility_label, volatility_pct = calculate_volatility(prices)

    record_prediction(coin_id, predicted, trend_price, momentum_price, crossover_price)
    news_items, news_error = get_news()
    fear_greed = get_fear_greed_index()
    sentiment_score = score_news_sentiment(news_items)
    search_interest = get_search_interest(AVAILABLE_COINS[coin_id])

    compare_prices = None
    compare_name = None
    if compare_id and compare_id != coin_id:
        try:
            _, compare_prices = get_historical_prices(compare_id, days=days)
            compare_name = AVAILABLE_COINS[compare_id]
        except (ValueError, requests.RequestException):
            compare_prices = None

    return render_template(
        "index.html",
        coin=AVAILABLE_COINS[coin_id],
        coin_id=coin_id,
        available_coins=AVAILABLE_COINS,
        dates=dates,
        prices=prices,
        current=round(current, 2),
        predicted=round(predicted, 2),
        percent_change=round(percent_change, 2),
        percent_change_abs=round(abs(percent_change), 2),
        change_24h=details.get("change_24h"),
        change_7d=details.get("change_7d"),
        last_updated=datetime.now().strftime("%I:%M %p"),
        days=days,
        volatility_label=volatility_label,
        volatility_pct=round(volatility_pct, 1),
        compare_id=compare_id,
        compare_name=compare_name,
        compare_prices=compare_prices,
        news_items=news_items,
        news_error=news_error,
        fear_greed=fear_greed,
        sentiment_score=round(sentiment_score, 2),
        search_interest=search_interest,
        trend_weight=round(trend_weight * 100),
        momentum_weight=round(momentum_weight * 100),
        crossover_weight=round(crossover_weight * 100),
    )


@app.route("/history")
def history():
    coin_id = request.args.get("coin", "bitcoin")
    if coin_id not in AVAILABLE_COINS:
        coin_id = "bitcoin"

    try:
        dates, prices = get_historical_prices(coin_id, days=30)
    except (ValueError, requests.RequestException) as e:
        return render_template(
            "error.html", message=str(e), coin_id=coin_id, available_coins=AVAILABLE_COINS
        )

    records = get_prediction_history(coin_id, dates, prices)
    return render_template(
        "history.html",
        coin=AVAILABLE_COINS[coin_id],
        coin_id=coin_id,
        available_coins=AVAILABLE_COINS,
        records=records,
    )


@app.route("/about")
def about():
    return render_template("about.html", available_coins=AVAILABLE_COINS)


@app.route("/portfolio")
def portfolio():
    try:
        prices = get_all_current_prices()
    except ValueError as e:
        return render_template(
            "error.html", message=str(e), coin_id="bitcoin", available_coins=AVAILABLE_COINS
        )

    return render_template(
        "portfolio.html",
        available_coins=AVAILABLE_COINS,
        prices=prices,
    )


if __name__ == "__main__":
    app.run(debug=True)
