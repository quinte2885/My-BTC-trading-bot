“””
BTC Smart DCA Bot — Full Suite + Market Regime Detection + Telegram Commands
“””

import json, os, csv, time, uuid, requests, schedule, sys
import numpy as np
from datetime import datetime
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend
import jwt

try:
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
ML_AVAILABLE = True
except ImportError:
ML_AVAILABLE = False
print(“scikit-learn not installed — AI layer disabled”)

sys.stdout.reconfigure(line_buffering=True)

# ─────────────────────────────────────────────

# CONFIGURATION

# ─────────────────────────────────────────────

BUY_AMOUNT_USD        = 10
MAX_BUY_AMOUNT        = 20
COMPOUND_RATE         = 0.20
PRODUCT_ID            = “BTC-USD”
LOG_FILE              = os.path.expanduser(”~/Desktop/dca_log.csv”)
PERF_LOG              = os.path.expanduser(”~/Desktop/performance_log.csv”)
PARAMS_FILE           = os.path.expanduser(”~/Desktop/bot_params.json”)
STATE_FILE            = os.path.expanduser(”~/Desktop/bot_state.json”)

TAKE_PROFIT_PCT       = 50.0
STOP_LOSS_PCT         = 8.0
TRAILING_STOP_PCT     = 10.0
TRAILING_ACTIVATES    = 3.0

VOLUME_HIGH_THRESHOLD = 150
VOLUME_LOW_THRESHOLD  = 50
VOLUME_LOOKBACK       = 20

REGIME_MA_DAYS        = 200
BULL_TAKE_PROFIT      = 100.0
BULL_STOP_LOSS        = 10.0
BEAR_TAKE_PROFIT      = 30.0
BEAR_STOP_LOSS        = 6.0
SIDEWAYS_TAKE_PROFIT  = 50.0
SIDEWAYS_STOP_LOSS    = 8.0

INTERVAL              = “daily”
SUMMARY_TIME          = “18:00”
DRY_RUN               = False
BASE_URL              = “https://api.coinbase.com”

# ─────────────────────────────────────────────

# AI LAYER

# ─────────────────────────────────────────────

AI_MODEL_FILE    = os.path.expanduser(”~/Desktop/ai_model.json”)
AI_MIN_TRADES    = 10   # Minimum trades before AI activates
AI_HIGH_SCORE    = 70   # Score above this = keep or upgrade decision
AI_LOW_SCORE     = 35   # Score below this = downgrade decision

ai_model = {
“trained”:        False,
“trade_features”: [],
“trade_outcomes”: [],
“feature_means”:  [],
“feature_stds”:   [],
“weights”:        [],
“total_scored”:   0,
“correct_calls”:  0,
}

def load_ai_model():
“”“Load saved AI model from disk.”””
global ai_model
if os.path.exists(AI_MODEL_FILE):
with open(AI_MODEL_FILE) as f:
ai_model = json.load(f)
if ai_model.get(“trained”):
print(f”AI model loaded — {len(ai_model[‘trade_features’])} trades trained”)
return True
return False

def save_ai_model():
“”“Save AI model to disk.”””
with open(AI_MODEL_FILE, “w”) as f:
json.dump(ai_model, f, indent=2)

def extract_features(daily_rsi, hourly_rsi, fg_value, vol_pct,
macro_sentiment, onchain_sentiment, price_vs_yesterday):
“”“Extract normalized feature vector from signal readings.”””
macro_score   = {“bullish”: 1, “neutral”: 0, “bearish”: -1}.get(macro_sentiment, 0)
onchain_score = {“bullish”: 1, “neutral”: 0, “bearish”: -1}.get(onchain_sentiment, 0)
price_above   = 1 if price_vs_yesterday and price_vs_yesterday == “above” else 0

```
return [
    daily_rsi   if daily_rsi   is not None else 50,
    hourly_rsi  if hourly_rsi  is not None else 50,
    fg_value    if fg_value    is not None else 50,
    vol_pct     if vol_pct     is not None else 100,
    macro_score,
    onchain_score,
    price_above,
]
```

def load_trade_history():
“”“Load trade history from dca_log.csv and extract features.”””
trades = []
if not os.path.exists(LOG_FILE):
return trades
with open(LOG_FILE) as f:
reader = csv.DictReader(f)
rows   = list(reader)

```
for i, row in enumerate(rows):
    action = row.get("Action", "")
    if action not in ("HALF_BUY", "FULL_BUY", "DOUBLE_BUY", "SKIP"):
        continue
    try:
        price  = float(row.get("BTC Price", 0))
        note   = row.get("Note", "")
        if price == 0:
            continue
        # Extract signals from note field
        daily_rsi   = 50.0
        hourly_rsi  = 50.0
        fg_val      = 25.0
        vol_pct     = 100.0

        # Look ahead to measure outcome (7 day price change)
        outcome = 0
        for j in range(i+1, min(i+8, len(rows))):
            future_price = float(rows[j].get("BTC Price", 0))
            if future_price > 0:
                pct_change = ((future_price - price) / price) * 100
                outcome    = 1 if pct_change > 0 else 0
                break

        features = [daily_rsi, hourly_rsi, fg_val, vol_pct, 0, 0, 0]
        trades.append({"features": features, "outcome": outcome, "price": price, "action": action})
    except Exception:
        continue
return trades
```

def train_ai_model():
“”“Train the AI model on historical trade data.”””
global ai_model
if not ML_AVAILABLE:
return False

```
trades = load_trade_history()
if len(trades) < AI_MIN_TRADES:
    print(f"  AI: Need {AI_MIN_TRADES} trades to train — have {len(trades)}")
    return False

features = [t["features"] for t in trades]
outcomes = [t["outcome"] for t in trades]

# Normalize features
features_arr = np.array(features, dtype=float)
means        = features_arr.mean(axis=0).tolist()
stds         = features_arr.std(axis=0).tolist()
stds         = [s if s > 0 else 1.0 for s in stds]

normalized = ((features_arr - np.array(means)) / np.array(stds)).tolist()

ai_model["trained"]        = True
ai_model["trade_features"] = normalized
ai_model["trade_outcomes"] = outcomes
ai_model["feature_means"]  = means
ai_model["feature_stds"]   = stds

save_ai_model()
print(f"  AI model trained on {len(trades)} trades")
return True
```

def score_entry(daily_rsi, hourly_rsi, fg_value, vol_pct,
macro_sentiment, onchain_sentiment, yesterday_close, current_price):
“”“Score a potential trade entry 0-100 using the AI model.”””
if not ML_AVAILABLE or not ai_model.get(“trained”):
return None, “AI not ready”

```
try:
    price_vs_yesterday = "above" if yesterday_close and current_price > yesterday_close else "below"
    features    = extract_features(daily_rsi, hourly_rsi, fg_value, vol_pct,
                                   macro_sentiment, onchain_sentiment, price_vs_yesterday)
    means       = ai_model["feature_means"]
    stds        = ai_model["feature_stds"]
    normalized  = [(f - m) / s for f, m, s in zip(features, means, stds)]

    # KNN similarity scoring
    stored      = ai_model["trade_features"]
    outcomes    = ai_model["trade_outcomes"]
    if not stored:
        return None, "No training data"

    distances = []
    for i, stored_feat in enumerate(stored):
        dist = sum((a - b) ** 2 for a, b in zip(normalized, stored_feat)) ** 0.5
        distances.append((dist, outcomes[i]))

    # Weight closer matches more heavily
    distances.sort(key=lambda x: x[0])
    top_k      = min(5, len(distances))
    weights    = [1.0 / (d[0] + 0.001) for d in distances[:top_k]]
    total_w    = sum(weights)
    score      = sum(w * o for w, o in zip(weights, [d[1] for d in distances[:top_k]])) / total_w
    score_pct  = round(score * 100)

    # Generate reasoning
    if daily_rsi and daily_rsi < 40:
        reason = "oversold daily RSI — historically strong entry"
    elif hourly_rsi and hourly_rsi < 20:
        reason = "extreme hourly dip — historically best entries"
    elif fg_value and fg_value < 15:
        reason = "extreme fear — historically excellent accumulation"
    elif daily_rsi and hourly_rsi and daily_rsi > 80 and hourly_rsi > 70:
        reason = "both RSIs overbought — historically weaker entries"
    elif macro_sentiment == "bullish" and onchain_sentiment == "bullish":
        reason = "macro + on-chain both bullish — strong confirmation"
    else:
        reason = "mixed signals — moderate confidence"

    return score_pct, reason

except Exception as e:
    return None, f"Scoring error: {e}"
```

def apply_ai_adjustment(base_amount, base_type, ai_score, current_base):
“”“Adjust buy amount based on AI score.”””
if ai_score is None:
return base_amount, base_type, “AI not active”

```
original_type   = base_type
original_amount = base_amount

if ai_score < AI_LOW_SCORE and base_amount > 0:
    # Low confidence — downgrade one level
    if base_type == "DOUBLE_BUY":
        base_amount = current_base
        base_type   = "FULL_BUY"
    elif base_type == "FULL_BUY":
        base_amount = current_base / 2
        base_type   = "HALF_BUY"
    elif base_type == "HALF_BUY":
        base_amount = current_base / 4
        base_type   = "HALF_BUY"
    adjustment = f"AI downgraded: {original_type} to {base_type} (score: {ai_score})"

elif ai_score >= AI_HIGH_SCORE and base_amount > 0:
    # High confidence — upgrade one level
    if base_type == "HALF_BUY":
        base_amount = current_base
        base_type   = "FULL_BUY"
    elif base_type == "FULL_BUY":
        base_amount = min(current_base * 2, 50)
        base_type   = "DOUBLE_BUY"
    adjustment = f"AI upgraded: {original_type} to {base_type} (score: {ai_score})"

else:
    adjustment = f"AI confirmed: {base_type} (score: {ai_score})"

return base_amount, base_type, adjustment
```

# Load AI model on startup

load_ai_model()
train_ai_model()

# ─────────────────────────────────────────────

# TELEGRAM

# ─────────────────────────────────────────────

TELEGRAM_TOKEN   = “8672200439:AAGLKLAp-XX3OzA6VxAa7-Xw_nXg4AV1lSk”
TELEGRAM_CHAT_ID = “7519828264”

def send_telegram(message):
try:
url  = f”https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage”
data = {“chat_id”: TELEGRAM_CHAT_ID, “text”: message, “parse_mode”: “Markdown”}
requests.post(url, data=data, timeout=10)
except Exception as e:
print(f”  Telegram failed: {e}”)

# ─────────────────────────────────────────────

# TELEGRAM COMMAND LISTENER

# ─────────────────────────────────────────────

last_update_id = 0

def check_telegram_commands():
global last_update_id
try:
url      = f”https://api.telegram.org/bot{TELEGRAM_TOKEN}/getUpdates?offset={last_update_id + 1}&timeout=5”
response = requests.get(url, timeout=10)
data     = response.json()
updates  = data.get(“result”, [])
for update in updates:
last_update_id = update[“update_id”]
msg     = update.get(“message”, {})
chat_id = str(msg.get(“chat”, {}).get(“id”, “”))
text    = msg.get(“text”, “”).strip().upper()
if chat_id != TELEGRAM_CHAT_ID:
continue
print(f”  Telegram command: {text}”)
if text == “YES”:       handle_yes()
elif text == “NO”:      handle_no()
elif text == “STATUS”:  handle_status()
elif text == “STOP”:    handle_stop()
elif text == “RESUME”:  handle_resume()
elif text == “HELP”:    handle_help()
else: send_telegram(“Unknown command: “ + text + “\nType HELP for list”)
except Exception:
pass

def handle_yes():
if not params.get(“regime_confirmed”, True):
regime   = params.get(“current_regime”, “sideways”)
tp_map   = {“bull”: BULL_TAKE_PROFIT, “bear”: BEAR_TAKE_PROFIT, “sideways”: SIDEWAYS_TAKE_PROFIT}
sl_map   = {“bull”: BULL_STOP_LOSS, “bear”: BEAR_STOP_LOSS, “sideways”: SIDEWAYS_STOP_LOSS}
params[“regime_confirmed”] = True
save_params(params)
send_telegram(“Regime Change Approved! “ + regime.upper() + “ mode\nTP: “ + str(tp_map[regime]) + “% | SL: “ + str(sl_map[regime]) + “%”)
print(”  Regime approved via Telegram”)
else:
send_telegram(“No pending regime change to approve.”)

def handle_no():
if not params.get(“regime_confirmed”, True):
params[“regime_confirmed”] = True
save_params(params)
send_telegram(“Regime Change Rejected\nBot keeps current settings\nTP: “ + str(TAKE_PROFIT_PCT) + “% | SL: “ + str(STOP_LOSS_PCT) + “%”)
print(”  Regime rejected via Telegram”)
else:
send_telegram(“No pending regime change to reject.”)

def handle_status():
avg       = state[“avg_buy_price”]
btc_held  = state[“total_btc”]
usd_spent = state[“total_usd_spent”]
regime    = params.get(“current_regime”, “sideways”)
win_rate  = 0
if params[“total_trades”] > 0:
win_rate = round((params[“winning_trades”] / params[“total_trades”]) * 100, 1)
status = “STOPPED” if params.get(“trading_stopped”, False) else “ACTIVE”
send_telegram(
“Bot Status: “ + status + “\n”
“Regime: “ + regime.upper() + “\n”
“TP: “ + str(TAKE_PROFIT_PCT) + “% | SL: “ + str(STOP_LOSS_PCT) + “%\n”
“BTC Held: “ + str(round(btc_held, 6)) + “\n”
“USD Spent: $” + str(round(usd_spent, 2)) + “\n”
“Avg Buy: $” + str(round(avg, 2)) + “\n”
“Trades: “ + str(params[“total_trades”]) + “ | Win Rate: “ + str(win_rate) + “%\n”
“Base Buy: $” + str(params[“current_buy_amount”]) + “\n”
“Compounded: $” + str(round(params[“total_compounded”], 2))
)

def handle_stop():
params[“trading_stopped”] = True
save_params(params)
send_telegram(“Trading STOPPED\nNo new trades will be placed\nType RESUME to restart”)
print(”  Trading stopped via Telegram”)

def handle_resume():
params[“trading_stopped”] = False
save_params(params)
send_telegram(“Trading RESUMED\nNext trade: tomorrow at 09:00 AM”)
print(”  Trading resumed via Telegram”)

def handle_help():
send_telegram(
“Available Commands:\n”
“YES - Approve regime change\n”
“NO - Reject regime change\n”
“STATUS - View position and settings\n”
“STOP - Emergency stop trading\n”
“RESUME - Resume trading\n”
“HELP - Show this message”
)

# ─────────────────────────────────────────────

# ADAPTIVE PARAMETERS

# ─────────────────────────────────────────────

DEFAULT_PARAMS = {
“rsi_oversold”:       40,
“rsi_overbought”:     80,
“ma_days”:            50,
“total_trades”:       0,
“winning_trades”:     0,
“losing_trades”:      0,
“last_adjusted”:      None,
“current_buy_amount”: 10.0,
“total_compounded”:   0.0,
“current_regime”:     “sideways”,
“regime_confirmed”:   True,
}

def load_params():
if os.path.exists(PARAMS_FILE):
with open(PARAMS_FILE) as f:
p = json.load(f)
# Add missing keys from defaults
for k, v in DEFAULT_PARAMS.items():
if k not in p:
p[k] = v
print(f”Loaded params — RSI: {p[‘rsi_oversold’]}/{p[‘rsi_overbought’]} | Regime: {p[‘current_regime’].upper()}”)
return p
save_params(DEFAULT_PARAMS.copy())
print(“Initialized default params”)
return DEFAULT_PARAMS.copy()

def save_params(p):
with open(PARAMS_FILE, “w”) as f:
json.dump(p, f, indent=2)

params = load_params()

# Apply regime settings on startup

def apply_regime_settings():
global TAKE_PROFIT_PCT, STOP_LOSS_PCT
regime = params.get(“current_regime”, “sideways”)
confirmed = params.get(“regime_confirmed”, True)
if not confirmed:
return  # Wait for user approval
if regime == “bull”:
TAKE_PROFIT_PCT = BULL_TAKE_PROFIT
STOP_LOSS_PCT   = BULL_STOP_LOSS
elif regime == “bear”:
TAKE_PROFIT_PCT = BEAR_TAKE_PROFIT
STOP_LOSS_PCT   = BEAR_STOP_LOSS
else:
TAKE_PROFIT_PCT = SIDEWAYS_TAKE_PROFIT
STOP_LOSS_PCT   = SIDEWAYS_STOP_LOSS

apply_regime_settings()

# ─────────────────────────────────────────────

# LOAD API KEYS

# ─────────────────────────────────────────────

def load_keys():
path = os.path.expanduser(”~/Desktop/coinbase_keys.json”)
if not os.path.exists(path):
print(“coinbase_keys.json not found.”)
exit(1)
with open(path) as f:
keys = json.load(f)
api_key    = keys.get(“name”)
api_secret = keys.get(“privateKey”)
if not api_key or not api_secret:
print(“Could not read keys.”)
exit(1)
print(“API keys loaded”)
return api_key, api_secret

API_KEY, API_SECRET = load_keys()

# ─────────────────────────────────────────────

# JWT AUTH

# ─────────────────────────────────────────────

def build_jwt(method, path):
private_key = serialization.load_pem_private_key(
API_SECRET.encode(“utf-8”), password=None, backend=default_backend()
)
payload = {
“sub”: API_KEY, “iss”: “cdp”,
“nbf”: int(time.time()), “exp”: int(time.time()) + 120,
“uri”: f”{method} api.coinbase.com{path}”,
}
return jwt.encode(payload, private_key, algorithm=“ES256”,
headers={“kid”: API_KEY, “nonce”: str(uuid.uuid4())})

def get_headers(method, path):
return {“Authorization”: f”Bearer {build_jwt(method, path)}”, “Content-Type”: “application/json”}

# ─────────────────────────────────────────────

# STATE

# ─────────────────────────────────────────────

DEFAULT_STATE = {
“total_btc”:            0.0,
“total_usd_spent”:      0.0,
“avg_buy_price”:        0.0,
“cycles”:               0,
“peak_price”:           0.0,
“entry_rsi”:            None,
“entry_fg”:             None,
“entry_ma_signal”:      None,
“entry_time”:           None,
“last_decision”:        “None”,
“opportunity_used_today”: False,
“opportunity_last_date”:  None,
}

def load_state():
if os.path.exists(STATE_FILE):
with open(STATE_FILE) as f:
s = json.load(f)
# Add any missing keys from defaults
for k, v in DEFAULT_STATE.items():
if k not in s:
s[k] = v
print(f”Loaded position state — BTC: {s[‘total_btc’]:.6f} | Avg: ${s[‘avg_buy_price’]:,.2f} | Spent: ${s[‘total_usd_spent’]:.2f}”)
return s
print(“No saved state found — starting fresh”)
return DEFAULT_STATE.copy()

def save_state():
with open(STATE_FILE, “w”) as f:
json.dump(state, f, indent=2)

state = load_state()

# ─────────────────────────────────────────────

# PRICE + HISTORICAL DATA

# ─────────────────────────────────────────────

def get_btc_price():
path     = f”/api/v3/brokerage/products/{PRODUCT_ID}”
response = requests.get(BASE_URL + path, headers=get_headers(“GET”, path))
return float(response.json()[“price”])

def get_historical_data(days=210):
url      = f”https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days={days}&interval=daily”
response = requests.get(url, timeout=10)
data     = response.json()
prices   = [float(p[1]) for p in data.get(“prices”, [])]
volumes  = [float(v[1]) for v in data.get(“total_volumes”, [])]
return prices, volumes

def get_hourly_prices():
url      = “https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=2&interval=hourly”
response = requests.get(url, timeout=10)
data     = response.json()
return [float(p[1]) for p in data.get(“prices”, [])]

# ─────────────────────────────────────────────

# MACRO DATA

# ─────────────────────────────────────────────

MACRO_HEADERS = {“User-Agent”: “Mozilla/5.0”}

def fetch_macro_data():
“”“Fetch DXY, S&P 500, and Gold from Yahoo Finance with data validation.”””
results = {}
symbols = {
“dxy”:  “DX-Y.NYB”,
“spy”:  “SPY”,
“gold”: “GC=F”,
}
for key, symbol in symbols.items():
try:
url      = f”https://query1.finance.yahoo.com/v8/finance/chart/{symbol}?interval=1d&range=5d”
r        = requests.get(url, headers=MACRO_HEADERS, timeout=10)
data     = r.json()
result   = data[“chart”][“result”][0]
closes   = result[“indicators”][“quote”][0][“close”]
closes   = [c for c in closes if c is not None and c > 0]
if len(closes) >= 2:
current = closes[-1]
prev    = closes[-2]
# Validate: reject if change is unrealistically large
change  = ((current - prev) / prev) * 100
if abs(change) > 20:
print(f”  ⚠️ {key} data rejected — unrealistic change: {change:.2f}%”)
results[key] = None
else:
results[key] = {“price”: round(current, 2), “change”: round(change, 2)}
else:
print(f”  ⚠️ {key} insufficient data — treating as neutral”)
results[key] = None
except Exception as e:
print(f”  ⚠️ {key} fetch failed — treating as neutral”)
results[key] = None
return results

def analyze_macro_score(macro):
“”“Score macro environment: positive = bullish, negative = bearish.”””
if not macro:
return 0, “neutral”
score = 0
details = []
dxy  = macro.get(“dxy”)
spy  = macro.get(“spy”)
gold = macro.get(“gold”)
if dxy:
if dxy[“change”] <= -0.5:
score += 1
details.append(f”DXY falling {dxy[‘change’]:+.2f}% (bullish)”)
elif dxy[“change”] >= 0.5:
score -= 1
details.append(f”DXY rising {dxy[‘change’]:+.2f}% (bearish)”)
if spy:
if spy[“change”] >= 1.0:
score += 1
details.append(f”S&P 500 up {spy[‘change’]:+.2f}% (bullish)”)
elif spy[“change”] <= -1.5:
score -= 1
details.append(f”S&P 500 down {spy[‘change’]:+.2f}% (bearish)”)
if gold:
if gold[“change”] >= 1.0:
score += 1
details.append(f”Gold up {gold[‘change’]:+.2f}% (bullish)”)
elif gold[“change”] <= -1.0:
score -= 1
details.append(f”Gold down {gold[‘change’]:+.2f}% (bearish)”)
if score >= 2:
sentiment = “bullish”
elif score <= -2:
sentiment = “bearish”
else:
sentiment = “neutral”
return score, sentiment, details

# ─────────────────────────────────────────────

# ON-CHAIN DATA

# ─────────────────────────────────────────────

onchain_state = {
“prev_hodling”:  None,
“prev_mempool”:  None,
“prev_hashrate”: None,
}

def fetch_onchain_data():
“”“Fetch on-chain data with validation — rejects zero or unrealistic values.”””
result = {}
try:
r    = requests.get(“https://api.blockchair.com/bitcoin/stats”, timeout=10)
data = r.json().get(“data”, {})

```
    # Validate each field — reject zeros and None
    mempool = data.get("mempool_transactions")
    if mempool is not None and mempool > 0:
        result["mempool_tx"] = mempool
    else:
        print("  ⚠️ Mempool data invalid — treating as neutral")

    hodling = data.get("hodling_addresses")
    if hodling is not None and hodling > 1000000:  # Must be > 1M to be valid
        result["hodling"] = hodling
    else:
        print("  ⚠️ Hodling addresses invalid — treating as neutral")

    avg_fee = data.get("average_transaction_fee_usd_24h")
    if avg_fee is not None and 0 < avg_fee < 1000:  # Reject unrealistic fees
        result["avg_fee_usd"] = avg_fee
    else:
        print("  ⚠️ Avg fee data invalid — treating as neutral")

    result["dominance"] = data.get("market_dominance_percentage")

except Exception as e:
    print(f"  ⚠️ Blockchair failed — on-chain treating as neutral")

try:
    r         = requests.get("https://mempool.space/api/v1/mining/hashrate/3m", timeout=10)
    data      = r.json()
    hashrates = data.get("hashrates", [])
    current_hr = data.get("currentHashrate", 0)

    if current_hr and current_hr > 0:
        result["current_hashrate"] = current_hr
        if len(hashrates) >= 30:
            result["avg_30d_hashrate"] = sum(h.get("avgHashrate", 0) for h in hashrates[-30:]) / 30
        else:
            result["avg_30d_hashrate"] = current_hr
    else:
        print("  ⚠️ Hashrate data invalid — treating as neutral")

except Exception as e:
    print(f"  ⚠️ Mempool failed — hashrate treating as neutral")

return result
```

def get_yesterday_close():
“”“Fetch yesterday’s BTC closing price from CoinGecko.”””
try:
url      = “https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=2&interval=daily”
r        = requests.get(url, timeout=10)
prices   = [p[1] for p in r.json().get(“prices”, []) if p[1] is not None]
if len(prices) >= 2:
return prices[-2]
return None
except Exception:
return None

def analyze_onchain_score(onchain):
“”“Score on-chain environment: positive = bullish, negative = bearish.”””
if not onchain:
return 0, “neutral”, []
score   = 0
details = []

```
# Mempool
mempool = onchain.get("mempool_tx")
if mempool is not None:
    if mempool < 10000:
        score += 1
        details.append(f"Low mempool {mempool:,} (bullish)")
    elif mempool > 150000:
        score -= 1
        details.append(f"High mempool {mempool:,} (bearish)")

# Hodling addresses change
hodling = onchain.get("hodling")
if hodling and onchain_state["prev_hodling"]:
    change = hodling - onchain_state["prev_hodling"]
    if change > 5000:
        score += 1
        details.append(f"Hodling addresses up +{change:,} (bullish)")
    elif change < -5000:
        score -= 1
        details.append(f"Hodling addresses down {change:,} (bearish)")
onchain_state["prev_hodling"] = hodling

# Hashrate vs 30d avg
hr      = onchain.get("current_hashrate", 0)
avg_hr  = onchain.get("avg_30d_hashrate", 0)
if hr and avg_hr and avg_hr > 0:
    hr_pct = ((hr - avg_hr) / avg_hr) * 100
    if hr_pct >= 5:
        score += 1
        details.append(f"Hashrate +{hr_pct:.1f}% vs 30d avg (bullish)")
    elif hr_pct <= -10:
        score -= 1
        details.append(f"Hashrate {hr_pct:.1f}% vs 30d avg (bearish)")

# Average fee
avg_fee = onchain.get("avg_fee_usd")
if avg_fee is not None:
    if avg_fee < 0.50:
        score += 1
        details.append(f"Low avg fee ${avg_fee:.2f} (bullish)")
    elif avg_fee > 5.0:
        score -= 1
        details.append(f"High avg fee ${avg_fee:.2f} (bearish)")

if score >= 2:
    sentiment = "bullish"
elif score <= -2:
    sentiment = "bearish"
else:
    sentiment = "neutral"

return score, sentiment, details
```

# ─────────────────────────────────────────────

# FEAR & GREED

# ─────────────────────────────────────────────

def get_fear_and_greed():
try:
response = requests.get(“https://api.alternative.me/fng/?limit=1”, timeout=10)
data     = response.json()
value    = int(data[“data”][0][“value”])
label    = data[“data”][0][“value_classification”]
return value, label
except Exception:
return None, “Unknown”

def fear_greed_signal(value):
if value is None: return “neutral”
if value <= 25:   return “extreme_fear”
if value <= 50:   return “fear”
if value <= 75:   return “greed”
return “extreme_greed”

# ─────────────────────────────────────────────

# TECHNICAL INDICATORS

# ─────────────────────────────────────────────

def calculate_rsi(prices, period=14):
if len(prices) < period + 1:
return None
gains, losses = [], []
for i in range(1, period + 1):
change = prices[-period + i] - prices[-period + i - 1]
gains.append(change if change > 0 else 0)
losses.append(abs(change) if change < 0 else 0)
avg_gain = sum(gains) / period
avg_loss = sum(losses) / period
if avg_loss == 0:
return 100
return round(100 - (100 / (1 + avg_gain / avg_loss)), 2)

def calculate_ma(prices, period=50):
if len(prices) < period:
return None
return sum(prices[-period:]) / period

def analyze_volume(volumes):
if len(volumes) < VOLUME_LOOKBACK + 1:
return “normal”, 100
avg_volume     = sum(volumes[-VOLUME_LOOKBACK:-1]) / VOLUME_LOOKBACK
current_volume = volumes[-1]
if avg_volume == 0:
return “normal”, 100
pct = round((current_volume / avg_volume) * 100, 1)
if pct >= VOLUME_HIGH_THRESHOLD:  return “high”, pct
elif pct <= VOLUME_LOW_THRESHOLD: return “low”, pct
return “normal”, pct

# ─────────────────────────────────────────────

# MARKET REGIME DETECTION

# ─────────────────────────────────────────────

def detect_regime(prices, current_price):
if len(prices) < REGIME_MA_DAYS:
return “sideways”, None
ma200     = sum(prices[-REGIME_MA_DAYS:]) / REGIME_MA_DAYS
pct_above = ((current_price - ma200) / ma200) * 100
if len(prices) >= 60:
recent_avg = sum(prices[-30:]) / 30
prior_avg  = sum(prices[-60:-30]) / 30
trend      = ((recent_avg - prior_avg) / prior_avg) * 100
else:
trend = 0
if current_price > ma200 and pct_above > 5 and trend > 0:
return “bull”, ma200
elif current_price < ma200 and pct_above < -5 and trend < 0:
return “bear”, ma200
return “sideways”, ma200

def check_regime_change(prices, current_price):
global TAKE_PROFIT_PCT, STOP_LOSS_PCT
new_regime, ma200 = detect_regime(prices, current_price)
old_regime        = params.get(“current_regime”, “sideways”)
confirmed         = params.get(“regime_confirmed”, True)
regime_emoji      = {“bull”: “🐂”, “bear”: “🐻”, “sideways”: “➡️”}
tp_map            = {“bull”: BULL_TAKE_PROFIT, “bear”: BEAR_TAKE_PROFIT, “sideways”: SIDEWAYS_TAKE_PROFIT}
sl_map            = {“bull”: BULL_STOP_LOSS,   “bear”: BEAR_STOP_LOSS,   “sideways”: SIDEWAYS_STOP_LOSS}

```
if new_regime != old_regime:
    print(f"  Regime change detected: {old_regime.upper()} to {new_regime.upper()}")
    ma_str = f"${ma200:,.2f}" if ma200 else "N/A"
    msg = (
        "🔄 *Market Regime Change Detected!*\n"
        f"{regime_emoji.get(old_regime,'➡️')} {old_regime.upper()} "
        f"to {regime_emoji.get(new_regime,'➡️')} {new_regime.upper()}\n\n"
        "*New settings if approved:*\n"
        f"Take Profit: {tp_map[new_regime]}%\n"
        f"Stop Loss: {sl_map[new_regime]}%\n"
        f"200-day MA: {ma_str}\n"
        f"Current Price: ${current_price:,.2f}\n\n"
        "Reply YES to apply new settings\n"
        "or bot keeps current settings until tomorrow"
    )
    send_telegram(msg)
    params["current_regime"]   = new_regime
    params["regime_confirmed"] = False
    save_params(params)
    print("  Telegram alert sent — awaiting approval")
else:
    if confirmed:
        if new_regime == "bull":
            TAKE_PROFIT_PCT = BULL_TAKE_PROFIT
            STOP_LOSS_PCT   = BULL_STOP_LOSS
        elif new_regime == "bear":
            TAKE_PROFIT_PCT = BEAR_TAKE_PROFIT
            STOP_LOSS_PCT   = BEAR_STOP_LOSS
        else:
            TAKE_PROFIT_PCT = SIDEWAYS_TAKE_PROFIT
            STOP_LOSS_PCT   = SIDEWAYS_STOP_LOSS
    emoji = regime_emoji.get(new_regime, "➡️")
    print(f"  {emoji} Regime: {new_regime.upper()} (TP: {tp_map[new_regime]}% | SL: {sl_map[new_regime]}%)")

return new_regime, ma200
```

# ─────────────────────────────────────────────

# SMART BUY DECISION

# ─────────────────────────────────────────────

def get_buy_amount(current_price, daily_prices, volumes, hourly_prices, fg_value, fg_label, macro=None, onchain=None, yesterday_close=None):
rsi        = calculate_rsi(daily_prices)
hourly_rsi = calculate_rsi(hourly_prices)
ma         = calculate_ma(daily_prices, params[“ma_days”])
fg         = fear_greed_signal(fg_value)
vol_signal, vol_pct = analyze_volume(volumes)
macro_score, macro_sentiment, macro_details   = analyze_macro_score(macro)
onchain_score, onchain_sentiment, onchain_details = analyze_onchain_score(onchain)

```
print(f"\n  📊 Technical Analysis:")

if rsi is None:
    print(f"  Daily RSI    : Not enough data")
    rsi_signal = "neutral"
else:
    print(f"  Daily RSI    : {rsi} ", end="")
    if rsi < params["rsi_oversold"]:
        print(f"(oversold ✅)")
        rsi_signal = "good"
    elif rsi > params["rsi_overbought"]:
        print(f"(overbought ❌)")
        rsi_signal = "bad"
    else:
        print("(neutral 🟡)")
        rsi_signal = "neutral"

if hourly_rsi is None:
    print(f"  Hourly RSI   : Not enough data")
    hourly_signal = "neutral"
else:
    print(f"  Hourly RSI   : {hourly_rsi} ", end="")
    if hourly_rsi < params["rsi_oversold"]:
        print(f"(oversold ✅)")
        hourly_signal = "good"
    elif hourly_rsi > params["rsi_overbought"]:
        print(f"(overbought ❌)")
        hourly_signal = "bad"
    else:
        print("(neutral 🟡)")
        hourly_signal = "neutral"

if ma is None:
    print(f"  50-day MA    : Not enough data")
    ma_signal = "neutral"
else:
    print(f"  50-day MA    : ${ma:,.2f} ", end="")
    if current_price > ma:
        print("(uptrend ✅)")
        ma_signal = "good"
    else:
        print("(downtrend ❌)")
        ma_signal = "bad"

fg_emoji = {"extreme_fear": "😱", "fear": "😟", "greed": "😏", "extreme_greed": "🤑"}.get(fg, "🤷")
print(f"  Fear & Greed : {fg_value} — {fg_label} {fg_emoji}")

vol_emoji = {"high": "📈", "low": "📉", "normal": "➡️"}.get(vol_signal, "➡️")
print(f"  Volume       : {vol_pct}% of 20-day avg {vol_emoji}")

# Macro display
macro_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(macro_sentiment, "🟡")
print(f"  Macro        : {macro_emoji} {macro_sentiment.upper()} (score: {macro_score})")
for d in macro_details:
    print(f"    - {d}")

# On-chain display
onchain_emoji = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(onchain_sentiment, "🟡")
print(f"  On-Chain     : {onchain_emoji} {onchain_sentiment.upper()} (score: {onchain_score})")
for d in onchain_details:
    print(f"    - {d}")

win_rate = 0
if params["total_trades"] > 0:
    win_rate = round((params["winning_trades"] / params["total_trades"]) * 100, 1)
print(f"  Win Rate     : {win_rate}% ({params['winning_trades']}W / {params['losing_trades']}L)")

current_base = params["current_buy_amount"]
print(f"  Base Buy     : ${current_base:.2f}")

good_technicals = rsi_signal == "good" and ma_signal == "good"
bad_technicals  = rsi_signal == "bad"  and ma_signal == "bad"

if good_technicals and fg == "extreme_fear":
    base_amount, base_type = min(current_base * 2, MAX_BUY_AMOUNT * 2), "DOUBLE_BUY"
elif good_technicals and fg == "fear":
    base_amount, base_type = current_base, "FULL_BUY"
elif bad_technicals and fg in ("extreme_greed", "greed"):
    base_amount, base_type = 0, "SKIP"
else:
    base_amount, base_type = current_base / 2, "HALF_BUY"

original_type = base_type
if vol_signal == "high" and base_amount > 0:
    if base_type == "HALF_BUY":   base_amount, base_type = current_base, "FULL_BUY"
    elif base_type == "FULL_BUY": base_amount, base_type = min(current_base * 2, MAX_BUY_AMOUNT * 2), "DOUBLE_BUY"
    if base_type != original_type:
        print(f"  Volume upgrade: {original_type} to {base_type}")
elif vol_signal == "low":
    if base_type == "DOUBLE_BUY": base_amount, base_type = current_base, "FULL_BUY"
    elif base_type == "FULL_BUY": base_amount, base_type = current_base / 2, "HALF_BUY"
    elif base_type == "HALF_BUY": base_amount, base_type = 0, "SKIP"
    if base_type != original_type:
        print(f"  Volume downgrade: {original_type} to {base_type}")

# Overbought context check
both_rsi_overbought = (rsi is not None and rsi > 70 and
                       hourly_rsi is not None and hourly_rsi > 70)
price_above_yesterday = (yesterday_close is not None and
                         current_price > yesterday_close)

if both_rsi_overbought:
    if price_above_yesterday:
        max_upgrades = 0
        print(f"  ⚠️ Both RSIs overbought + price above yesterday → upgrades locked")
    else:
        max_upgrades = 1
        print(f"  ⚠️ Both RSIs overbought but price below yesterday → max 1 upgrade")
else:
    max_upgrades = 2

upgrades_used = 0

# Macro adjustment
pre_macro_type = base_type
if macro_sentiment == "bullish" and base_amount > 0 and upgrades_used < max_upgrades:
    if base_type == "HALF_BUY":
        base_amount, base_type = current_base, "FULL_BUY"
        upgrades_used += 1
    elif base_type == "FULL_BUY":
        base_amount, base_type = min(current_base * 2, MAX_BUY_AMOUNT * 2), "DOUBLE_BUY"
        upgrades_used += 1
    if base_type != pre_macro_type:
        print(f"  Macro upgrade: {pre_macro_type} to {base_type}")
elif macro_sentiment == "bearish":
    if base_type == "DOUBLE_BUY":
        base_amount, base_type = current_base, "FULL_BUY"
    elif base_type == "FULL_BUY":
        base_amount, base_type = current_base / 2, "HALF_BUY"
    elif base_type == "HALF_BUY":
        base_amount, base_type = 0, "SKIP"
    if base_type != pre_macro_type:
        print(f"  Macro downgrade: {pre_macro_type} to {base_type}")

# On-chain adjustment
pre_onchain_type = base_type
if onchain_sentiment == "bullish" and base_amount > 0 and upgrades_used < max_upgrades:
    if base_type == "HALF_BUY":
        base_amount, base_type = current_base, "FULL_BUY"
        upgrades_used += 1
    elif base_type == "FULL_BUY":
        base_amount, base_type = min(current_base * 2, MAX_BUY_AMOUNT * 2), "DOUBLE_BUY"
        upgrades_used += 1
    if base_type != pre_onchain_type:
        print(f"  On-chain upgrade: {pre_onchain_type} to {base_type}")
elif onchain_sentiment == "bearish":
    if base_type == "DOUBLE_BUY":
        base_amount, base_type = current_base, "FULL_BUY"
    elif base_type == "FULL_BUY":
        base_amount, base_type = current_base / 2, "HALF_BUY"
    elif base_type == "HALF_BUY":
        base_amount, base_type = 0, "SKIP"
    if base_type != pre_onchain_type:
        print(f"  On-chain downgrade: {pre_onchain_type} to {base_type}")

# Confidence score
bullish_count = 0
bearish_count = 0
total_signals = 0

# RSI signals
if rsi is not None:
    total_signals += 1
    if rsi_signal == "good":   bullish_count += 1
    elif rsi_signal == "bad":  bearish_count += 1

if hourly_rsi is not None:
    total_signals += 1
    if hourly_signal == "good":  bullish_count += 1
    elif hourly_signal == "bad": bearish_count += 1

# MA signal
total_signals += 1
if ma_signal == "good":  bullish_count += 1
elif ma_signal == "bad": bearish_count += 1

# F&G signal
if fg_value is not None:
    total_signals += 1
    if fg in ("extreme_fear", "fear"):   bullish_count += 1
    elif fg in ("extreme_greed", "greed"): bearish_count += 1

# Volume signal
total_signals += 1
if vol_signal == "high":  bullish_count += 1
elif vol_signal == "low": bearish_count += 1

# Macro signal
total_signals += 1
if macro_sentiment == "bullish":   bullish_count += 1
elif macro_sentiment == "bearish": bearish_count += 1

# On-chain signal
total_signals += 1
if onchain_sentiment == "bullish":   bullish_count += 1
elif onchain_sentiment == "bearish": bearish_count += 1

confidence = round((bullish_count / total_signals) * 100) if total_signals > 0 else 0
conflict   = bearish_count > 0 and bullish_count > 0

conf_emoji = "🟢" if confidence >= 70 else "🟡" if confidence >= 40 else "🔴"
print(f"\n  {conf_emoji} Confidence: {confidence}% bullish ({bullish_count} bull / {bearish_count} bear / {total_signals - bullish_count - bearish_count} neutral)")
if conflict:
    print(f"  ⚠️ Conflicting signals detected — proceed with caution")

# AI scoring and adjustment
ai_score, ai_reason     = score_entry(rsi, hourly_rsi, fg_value, vol_pct,
                                      macro_sentiment, onchain_sentiment,
                                      yesterday_close, current_price)
pre_ai_type             = base_type
base_amount, base_type, ai_adjustment = apply_ai_adjustment(
    base_amount, base_type, ai_score, current_base
)

ai_emoji = "🤖"
if ai_score is not None:
    ai_color = "🟢" if ai_score >= AI_HIGH_SCORE else "🔴" if ai_score < AI_LOW_SCORE else "🟡"
    print(f"  {ai_emoji} AI Score: {ai_color} {ai_score}/100 — {ai_reason}")
    print(f"  {ai_adjustment}")
else:
    print(f"  {ai_emoji} AI: not enough data yet — needs {AI_MIN_TRADES} trades")

decision_emoji = {"DOUBLE_BUY": "🔵", "FULL_BUY": "🟢", "HALF_BUY": "🟡", "SKIP": "🔴"}.get(base_type, "🟡")
print(f"  {decision_emoji} Final Decision: {base_type} — ${base_amount}")
state["last_decision"] = f"{base_type} ${base_amount}"
return base_amount, base_type, rsi, ma_signal, hourly_rsi, vol_pct, macro_sentiment, onchain_sentiment, confidence, bullish_count, bearish_count, ai_score, ai_reason, ai_adjustment
```

# ─────────────────────────────────────────────

# ORDER PLACEMENT

# ─────────────────────────────────────────────

def place_market_buy(usd_amount):
path  = “/api/v3/brokerage/orders”
order = {
“client_order_id”: f”dca-buy-{int(time.time())}”,
“product_id”: PRODUCT_ID, “side”: “BUY”,
“order_configuration”: {“market_market_ioc”: {“quote_size”: str(usd_amount)}}
}
return requests.post(BASE_URL + path, headers=get_headers(“POST”, path), data=json.dumps(order)).json()

def place_market_sell(btc_amount):
path  = “/api/v3/brokerage/orders”
order = {
“client_order_id”: f”dca-sell-{int(time.time())}”,
“product_id”: PRODUCT_ID, “side”: “SELL”,
“order_configuration”: {“market_market_ioc”: {“base_size”: f”{btc_amount:.8f}”}}
}
return requests.post(BASE_URL + path, headers=get_headers(“POST”, path), data=json.dumps(order)).json()

# ─────────────────────────────────────────────

# LOGGING

# ─────────────────────────────────────────────

def log_trade(action, usd_amount, btc_price, btc_amount, pnl=””, note=””):
file_exists = os.path.isfile(LOG_FILE)
with open(LOG_FILE, “a”, newline=””) as f:
writer = csv.writer(f)
if not file_exists:
writer.writerow([“Timestamp”, “Action”, “USD”, “BTC Price”, “BTC Amount”, “P&L (USD)”, “Note”])
writer.writerow([datetime.now().strftime(”%Y-%m-%d %H:%M:%S”), action, usd_amount,
round(btc_price, 2), round(btc_amount, 8), pnl, note])

def log_performance(result, pnl, entry_rsi, entry_fg, entry_ma, hold_days):
file_exists = os.path.isfile(PERF_LOG)
with open(PERF_LOG, “a”, newline=””) as f:
writer = csv.writer(f)
if not file_exists:
writer.writerow([“Timestamp”, “Result”, “P&L”, “Entry RSI”, “Entry F&G”,
“Entry MA”, “Hold Days”, “RSI Oversold”, “RSI Overbought”, “Regime”])
writer.writerow([datetime.now().strftime(”%Y-%m-%d %H:%M:%S”), result, round(pnl, 2),
entry_rsi, entry_fg, entry_ma, hold_days,
params[“rsi_oversold”], params[“rsi_overbought”],
params.get(“current_regime”, “sideways”)])

# ─────────────────────────────────────────────

# SELF LEARNING

# ─────────────────────────────────────────────

def adjust_params():
total = params[“total_trades”]
if total == 0 or total % 10 != 0:
return
win_rate = params[“winning_trades”] / total
print(f”\n  🧠 Self-Learning — {total} trades | Win rate: {round(win_rate*100,1)}%”)
adjusted = False
if win_rate < 0.4:
old_o, old_ob = params[“rsi_oversold”], params[“rsi_overbought”]
params[“rsi_oversold”]   = max(25, params[“rsi_oversold”] - 5)
params[“rsi_overbought”] = min(80, params[“rsi_overbought”] + 5)
send_telegram(f”🧠 *Bot Self-Adjusted*\nWin rate low ({round(win_rate*100,1)}%)\nRSI: {old_o}/{old_ob} to {params[‘rsi_oversold’]}/{params[‘rsi_overbought’]}”)
adjusted = True
elif win_rate > 0.65:
old_o, old_ob = params[“rsi_oversold”], params[“rsi_overbought”]
params[“rsi_oversold”]   = min(45, params[“rsi_oversold”] + 3)
params[“rsi_overbought”] = max(65, params[“rsi_overbought”] - 3)
send_telegram(f”🧠 *Bot Self-Adjusted*\nWin rate high ({round(win_rate*100,1)}%)\nRSI: {old_o}/{old_ob} to {params[‘rsi_oversold’]}/{params[‘rsi_overbought’]}”)
adjusted = True
else:
print(f”  Win rate acceptable — no adjustment”)
if adjusted:
params[“last_adjusted”] = datetime.now().strftime(”%Y-%m-%d %H:%M:%S”)
save_params(params)

# ─────────────────────────────────────────────

# SELL LOGIC

# ─────────────────────────────────────────────

def check_and_sell(current_price):
if state[“total_btc”] == 0:
return
avg    = state[“avg_buy_price”]
change = ((current_price - avg) / avg) * 100
if current_price > state[“peak_price”]:
state[“peak_price”] = current_price
save_state()
take_profit_price     = avg * (1 + TAKE_PROFIT_PCT / 100)
trailing_activates_at = avg * (1 + TRAILING_ACTIVATES / 100)
trailing_active       = state[“peak_price”] >= trailing_activates_at
if trailing_active:
stop_loss_price = state[“peak_price”] * (1 - TRAILING_STOP_PCT / 100)
stop_label      = f”Trailing Stop (-{TRAILING_STOP_PCT}% from peak)”
else:
stop_loss_price = avg * (1 - STOP_LOSS_PCT / 100)
stop_label      = f”Fixed Stop Loss (-{STOP_LOSS_PCT}%)”
print(f”\n  💰 Position Check:”)
print(f”  Avg Buy    : ${avg:,.2f}”)
print(f”  Peak       : ${state[‘peak_price’]:,.2f}”)
print(f”  Current    : ${current_price:,.2f} ({change:+.2f}%)”)
print(f”  Take Profit: ${take_profit_price:,.2f} (+{TAKE_PROFIT_PCT}%)”)
print(f”  {stop_label}: ${stop_loss_price:,.2f}”)
if not trailing_active:
print(f”  Trailing activates at: ${trailing_activates_at:,.2f} (+{TRAILING_ACTIVATES}%)”)
reason = None
if current_price >= take_profit_price:
reason = “TAKE_PROFIT”
elif current_price <= stop_loss_price:
reason = “TRAILING_STOP” if trailing_active else “STOP_LOSS”
if reason:
usd_value = state[“total_btc”] * current_price
pnl       = usd_value - state[“total_usd_spent”]
result    = “WIN” if pnl > 0 else “LOSS”
hold_days = 0
if state[“entry_time”]:
entry_dt  = datetime.strptime(state[“entry_time”], “%Y-%m-%d %H:%M:%S”)
hold_days = (datetime.now() - entry_dt).days
print(f”\n  {reason} triggered! {result} | P&L: ${pnl:+.2f}”)
emoji = “🏆” if result == “WIN” else “📉”
send_telegram(
f”{emoji} *{reason.replace(’_’,’ ’)} triggered!*\n”
f”Result: *{result}* | P&L: ${pnl:+.2f}\n”
f”BTC sold: {state[‘total_btc’]:.6f}\n”
f”Sale value: ${usd_value:.2f}\n”
f”Held: {hold_days} days\n”
f”Regime: {params.get(‘current_regime’,‘sideways’).upper()}\n”
f”Mode: {‘DRY RUN’ if DRY_RUN else ‘LIVE’}”
)
if DRY_RUN:
log_trade(f”DRY_{reason}”, round(usd_value, 2), current_price, state[“total_btc”], round(pnl, 2), “Simulated”)
else:
res = place_market_sell(state[“total_btc”])
if res.get(“success”):
log_trade(reason, round(usd_value, 2), current_price, state[“total_btc”], round(pnl, 2))
else:
error = res.get(“error_response”, {}).get(“message”, “Unknown”)
send_telegram(f”Sell FAILED: {error}”)
return
log_performance(result, pnl, state[“entry_rsi”], state[“entry_fg”], state[“entry_ma_signal”], hold_days)
params[“total_trades”] += 1
if pnl > 0:
params[“winning_trades”] += 1
reinvest = round(pnl * COMPOUND_RATE, 2)
old_buy  = params[“current_buy_amount”]
new_buy  = min(round(old_buy + reinvest, 2), MAX_BUY_AMOUNT)
params[“current_buy_amount”] = new_buy
params[“total_compounded”]  += reinvest
print(f”  Compounding: ${reinvest:.2f} reinvested — base buy ${old_buy:.2f} to ${new_buy:.2f}”)
comp_msg = (
“Compounding! “
f”Profit: ${pnl:.2f} | “
f”Reinvested: ${reinvest:.2f} (20%) | “
f”Base buy: ${old_buy:.2f} to ${new_buy:.2f} | “
f”Total: ${params[‘total_compounded’]:.2f}”
)
send_telegram(comp_msg)
else:
params[“losing_trades”] += 1
save_params(params)
adjust_params()
state.update({“total_btc”: 0.0, “total_usd_spent”: 0.0, “avg_buy_price”: 0.0,
“cycles”: 0, “peak_price”: 0.0, “entry_rsi”: None, “entry_fg”: None,
“entry_ma_signal”: None, “entry_time”: None})
save_state()
print(”  Position reset.”)
else:
print(”  No sell triggered. Continuing to accumulate.”)

# ─────────────────────────────────────────────

# DAILY SUMMARY

# ─────────────────────────────────────────────

def send_daily_summary():
try:
current_price  = get_btc_price()
avg            = state[“avg_buy_price”]
btc_held       = state[“total_btc”]
usd_spent      = state[“total_usd_spent”]
current_value  = btc_held * current_price
unrealized_pnl = current_value - usd_spent if btc_held > 0 else 0
pnl_pct        = ((current_price - avg) / avg * 100) if avg > 0 else 0
win_rate       = 0
if params[“total_trades”] > 0:
win_rate = round((params[“winning_trades”] / params[“total_trades”]) * 100, 1)
take_profit_price = avg * (1 + TAKE_PROFIT_PCT / 100) if avg > 0 else 0
stop_loss_price   = avg * (1 - STOP_LOSS_PCT / 100) if avg > 0 else 0
fg_value, fg_label = get_fear_and_greed()
fg_emoji  = {“Extreme Fear”: “😱”, “Fear”: “😟”, “Greed”: “😏”, “Extreme Greed”: “🤑”}.get(fg_label, “🤷”)
regime    = params.get(“current_regime”, “sideways”)
re_emoji  = {“bull”: “🐂”, “bear”: “🐻”, “sideways”: “➡️”}.get(regime, “➡️”)
msg = (
“📊 *Daily Summary*\n”
f”🕕 {datetime.now().strftime(’%B %d, %Y’)}\n\n”
“*Market*\n”
f”BTC: ${current_price:,.2f}\n”
f”F&G: {fg_value} — {fg_label} {fg_emoji}\n”
f”Regime: {re_emoji} {regime.upper()} (TP: {TAKE_PROFIT_PCT}% | SL: {STOP_LOSS_PCT}%)\n\n”
“*Position*\n”
f”BTC Held: {btc_held:.6f}\n”
f”USD Spent: ${usd_spent:.2f}\n”
f”Current Value: ${current_value:.2f}\n”
f”Unrealized P&L: ${unrealized_pnl:+.2f} ({pnl_pct:+.2f}%)\n\n”
“*Targets*\n”
f”Take Profit: ${take_profit_price:,.2f}\n”
f”Stop Loss: ${stop_loss_price:,.2f}\n\n”
“*Performance*\n”
f”Trades: {params[‘total_trades’]} | Win Rate: {win_rate}%\n”
f”Base Buy: ${params[‘current_buy_amount’]:.2f} | Compounded: ${params[‘total_compounded’]:.2f}\n”
f”Today: {state[‘last_decision’]}\n”
f”Mode: {‘DRY RUN’ if DRY_RUN else ‘LIVE’}”
)
send_telegram(msg)
print(f”\n  📊 Daily summary sent”)
except Exception as e:
print(f”  Summary failed: {e}”)

# ─────────────────────────────────────────────

# MAIN DCA CYCLE

# ─────────────────────────────────────────────

def run_dca():
print(f”\n[{datetime.now().strftime(’%Y-%m-%d %H:%M:%S’)}] ── DCA Cycle ──”)

```
# Check for emergency stop
if params.get("trading_stopped", False):
    print("  Trading is stopped — skipping cycle")
    send_telegram("⏸ Trade skipped — bot is stopped\nType RESUME to restart")
    return

try:
    current_price          = get_btc_price()
    print(f"  BTC Price : ${current_price:,.2f}")
    daily_prices, volumes  = get_historical_data(days=210)
    hourly_prices          = get_hourly_prices()
    fg_value, fg_label     = get_fear_and_greed()
    macro_data             = fetch_macro_data()
    onchain_data           = fetch_onchain_data()
    yesterday_close        = get_yesterday_close()

    # Check market regime
    current_regime, ma200  = check_regime_change(daily_prices, current_price)

    if yesterday_close:
        price_context = "above" if current_price > yesterday_close else "below"
        print(f"  Yesterday close: ${yesterday_close:,.2f} — today is {price_context}")

    buy_amount, buy_type, daily_rsi, ma_signal, hourly_rsi, vol_pct, macro_sentiment, onchain_sentiment, confidence, bull_count, bear_count, ai_score, ai_reason, ai_adjustment = get_buy_amount(
        current_price, daily_prices, volumes, hourly_prices, fg_value, fg_label, macro_data, onchain_data, yesterday_close
    )

    if buy_amount == 0:
        print("  Skipping cycle.")
        log_trade("SKIP", 0, current_price, 0, "", f"F&G:{fg_value} Vol:{vol_pct}%")
        send_telegram(
            f"⏭ *Buy Skipped*\n"
            f"BTC: ${current_price:,.2f}\n"
            f"RSI: {daily_rsi} | F&G: {fg_value} — {fg_label}\n"
            f"Volume: {vol_pct}% | Regime: {current_regime.upper()}"
        )
    else:
        btc_to_buy = buy_amount / current_price
        print(f"\n  Buying : ${buy_amount} to ~{btc_to_buy:.6f} BTC")
        if DRY_RUN:
            log_trade(f"DRY_{buy_type}", buy_amount, current_price, btc_to_buy,
                      "", f"F&G:{fg_value} Vol:{vol_pct}%")
            success = True
            print("  [DRY RUN] Simulating buy.")
        else:
            result  = place_market_buy(buy_amount)
            success = result.get("success", False)
            if success:
                log_trade(buy_type, buy_amount, current_price, btc_to_buy,
                          "", f"F&G:{fg_value} Vol:{vol_pct}% Regime:{current_regime}")
                print("  Buy placed!")
            else:
                error = result.get("error_response", {}).get("message", "Unknown")
                log_trade("BUY_FAILED", buy_amount, current_price, 0, "", error)
                send_telegram(f"Buy FAILED: {error}")
                success = False
        if success:
            state["total_btc"]       += btc_to_buy
            state["total_usd_spent"] += buy_amount
            state["avg_buy_price"]    = state["total_usd_spent"] / state["total_btc"]
            state["cycles"]          += 1
            if state["cycles"] == 1:
                state["entry_rsi"]       = daily_rsi
                state["entry_fg"]        = fg_value
                state["entry_ma_signal"] = ma_signal
                state["entry_time"]      = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            save_state()
            buy_emoji = {"DOUBLE_BUY": "🔵", "FULL_BUY": "🟢", "HALF_BUY": "🟡"}.get(buy_type, "🟡")
            re_emoji  = {"bull": "🐂", "bear": "🐻", "sideways": "➡️"}.get(current_regime, "➡️")
            macro_emoji_tg   = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(macro_sentiment, "🟡")
            onchain_emoji_tg = {"bullish": "🟢", "bearish": "🔴", "neutral": "🟡"}.get(onchain_sentiment, "🟡")
            conf_emoji_tg    = "🟢" if confidence >= 70 else "🟡" if confidence >= 40 else "🔴"
            conflict_note    = " Conflicting signals" if bear_count > 0 and bull_count > 0 else ""
            ai_score_str     = f"{ai_score}/100" if ai_score is not None else "training"
            ai_color_tg      = "🟢" if ai_score and ai_score >= AI_HIGH_SCORE else "🔴" if ai_score and ai_score < AI_LOW_SCORE else "🟡"
            send_telegram(
                f"{buy_emoji} *{'[DRY RUN] ' if DRY_RUN else ''}BTC {buy_type.replace('_',' ')}*\n"
                f"Spent: ${buy_amount} | BTC: ~{btc_to_buy:.6f}\n"
                f"Price: ${current_price:,.2f}\n"
                f"Daily RSI: {daily_rsi} | Hourly RSI: {hourly_rsi}\n"
                f"F&G: {fg_value} — {fg_label} | Volume: {vol_pct}%\n"
                f"Macro: {macro_emoji_tg} {macro_sentiment.upper()} | On-Chain: {onchain_emoji_tg} {onchain_sentiment.upper()}\n"
                f"Confidence: {conf_emoji_tg} {confidence}% ({bull_count} bull / {bear_count} bear){conflict_note}\n"
                f"AI Score: {ai_color_tg} {ai_score_str} — {ai_reason}\n"
                f"Regime: {re_emoji} {current_regime.upper()}\n"
                f"Avg buy: ${state['avg_buy_price']:,.2f}\n"
                f"Take profit: ${state['avg_buy_price']*(1+TAKE_PROFIT_PCT/100):,.2f} | "
                f"Stop: ${state['avg_buy_price']*(1-STOP_LOSS_PCT/100):,.2f}"
            )
            print(f"\n  Position: {state['total_btc']:.6f} BTC | "
                  f"Avg: ${state['avg_buy_price']:,.2f} | Cycles: {state['cycles']}")
            check_and_sell(current_price)
except Exception as e:
    print(f"  Error: {e}")
    log_trade("ERROR", 0, 0, 0, "", str(e))
    send_telegram(f"Bot Error: {str(e)}")
```

# ─────────────────────────────────────────────

# OPPORTUNITY BUY

# ─────────────────────────────────────────────

MIN_BALANCE_FOR_OPPORTUNITY = 300  # Minimum USD balance required
OPPORTUNITY_RSI_THRESHOLD   = 20   # Hourly RSI must be below this
OPPORTUNITY_FG_THRESHOLD    = 50   # F&G must be below this (fearful)
OPPORTUNITY_START_HOUR      = 12   # Only after 12:00 PM Mountain Time

def get_coinbase_balance():
“”“Fetch available USD balance from Coinbase.”””
try:
path     = “/api/v3/brokerage/accounts”
response = requests.get(BASE_URL + path, headers=get_headers(“GET”, path))
accounts = response.json().get(“accounts”, [])
for account in accounts:
if account.get(“currency”) == “USD”:
return float(account.get(“available_balance”, {}).get(“value”, 0))
return 0
except Exception as e:
print(f”  Balance check failed: {e}”)
return 0

def reset_opportunity_flag():
“”“Reset opportunity buy flag at midnight.”””
today = datetime.now().strftime(”%Y-%m-%d”)
if state.get(“opportunity_last_date”) != today:
state[“opportunity_used_today”] = False
state[“opportunity_last_date”]  = today
save_state()
print(f”  Opportunity buy flag reset for {today}”)

def check_opportunity_buy():
“”“Check if conditions are met for an intraday opportunity buy.”””
reset_opportunity_flag()

```
now = datetime.now()

# Safety check 1 — Only after 12:00 PM
if now.hour < OPPORTUNITY_START_HOUR:
    return

# Safety check 2 — Only one per day
if state.get("opportunity_used_today", False):
    return

print(f"  Opportunity Check [{now.strftime(chr(37)+"Y-%m-%d %H:%M:%S")}]")


try:
    # Fetch hourly RSI
    hourly_prices = get_hourly_prices()
    hourly_rsi    = calculate_rsi(hourly_prices)

    if hourly_rsi is None:
        print(f"  Hourly RSI unavailable")
        return

    print(f"  Hourly RSI: {hourly_rsi}")

    # Safety check 3 — Hourly RSI below threshold
    if hourly_rsi >= OPPORTUNITY_RSI_THRESHOLD:
        print(f"  RSI {hourly_rsi} above threshold {OPPORTUNITY_RSI_THRESHOLD} — no opportunity")
        return

    # Safety check 4 — F&G below 50 (fearful market)
    fg_value, fg_label = get_fear_and_greed()
    if fg_value and fg_value >= OPPORTUNITY_FG_THRESHOLD:
        print(f"  F&G {fg_value} above threshold {OPPORTUNITY_FG_THRESHOLD} — market not fearful enough")
        return

    # Safety check 5 — Not a BEAR regime
    if params.get("current_regime") == "bear":
        print(f"  Bear regime — skipping opportunity buy")
        return

    # Safety check 6 — Balance above minimum
    balance = get_coinbase_balance()
    if balance < MIN_BALANCE_FOR_OPPORTUNITY:
        print(f"  Balance ${balance:.2f} below minimum ${MIN_BALANCE_FOR_OPPORTUNITY} — skipping")
        send_telegram(
            f"⚠️ Opportunity buy skipped\n"
            f"Hourly RSI: {hourly_rsi} — would have triggered\n"
            f"Reason: Balance ${balance:.2f} below ${MIN_BALANCE_FOR_OPPORTUNITY} minimum"
        )
        return

    # All safety checks passed — execute opportunity buy
    current_price  = get_btc_price()
    buy_amount     = params["current_buy_amount"]
    btc_to_buy     = buy_amount / current_price

    print(f"  ✅ All safety checks passed — executing opportunity buy!")
    print(f"  Hourly RSI: {hourly_rsi} | F&G: {fg_value} | Balance: ${balance:.2f}")
    print(f"  Buying ${buy_amount} at ${current_price:,.2f}")

    if DRY_RUN:
        log_trade("DRY_OPPORTUNITY", buy_amount, current_price, btc_to_buy,
                 "", f"Hourly RSI:{hourly_rsi} F&G:{fg_value}")
        success = True
        print(f"  [DRY RUN] Simulating opportunity buy")
    else:
        result  = place_market_buy(buy_amount)
        success = result.get("success", False)
        if success:
            log_trade("OPPORTUNITY_BUY", buy_amount, current_price, btc_to_buy,
                     "", f"Hourly RSI:{hourly_rsi} F&G:{fg_value}")
            print(f"  ✅ Opportunity buy placed!")
        else:
            error = result.get("error_response", {}).get("message", "Unknown")
            send_telegram(f"Opportunity buy FAILED: {error}")
            return

    if success:
        state["total_btc"]            += btc_to_buy
        state["total_usd_spent"]      += buy_amount
        state["avg_buy_price"]         = state["total_usd_spent"] / state["total_btc"]
        state["opportunity_used_today"] = True
        state["opportunity_last_date"]  = now.strftime("%Y-%m-%d")
        save_state()

        send_telegram(
            f"🎯 *OPPORTUNITY BUY TRIGGERED*\n"
            f"Hourly RSI: {hourly_rsi} — Extreme oversold\n"
            f"BTC Price: ${current_price:,.2f}\n"
            f"Spent: ${buy_amount} | BTC: ~{btc_to_buy:.6f}\n"
            f"F&G: {fg_value} — {fg_label}\n"
            f"Balance check: ${balance:.2f} ✅\n"
            f"Regime: {params.get('current_regime','sideways').upper()} ✅\n"
            f"All 6 safety conditions passed ✅\n"
            f"Avg buy: ${state['avg_buy_price']:,.2f}"
        )

except Exception as e:
    print(f"  Opportunity check error: {e}")
```

# ─────────────────────────────────────────────

# SCHEDULER

# ─────────────────────────────────────────────

def start():
regime    = params.get(“current_regime”, “sideways”)
re_emoji  = {“bull”: “🐂”, “bear”: “🐻”, “sideways”: “➡️”}.get(regime, “➡️”)
print(”=” * 58)
print(”  BTC Smart DCA Bot — Full Suite + Regime Detection”)
print(f”  Mode     : {‘DRY RUN’ if DRY_RUN else ‘LIVE TRADING’}”)
print(f”  Regime   : {re_emoji} {regime.upper()} (TP: {TAKE_PROFIT_PCT}% | SL: {STOP_LOSS_PCT}%)”)
print(f”  Base Buy : ${params[‘current_buy_amount’]:.2f} (max ${MAX_BUY_AMOUNT})”)
print(f”  Bull     : TP {BULL_TAKE_PROFIT}% / SL {BULL_STOP_LOSS}%”)
print(f”  Sideways : TP {SIDEWAYS_TAKE_PROFIT}% / SL {SIDEWAYS_STOP_LOSS}%”)
print(f”  Bear     : TP {BEAR_TAKE_PROFIT}% / SL {BEAR_STOP_LOSS}%”)
print(f”  Summary  : Daily at {SUMMARY_TIME}”)
print(”=” * 58)
send_telegram(
f”🤖 *BTC DCA Bot Started — Full Suite*\n”
f”Mode: {‘DRY RUN’ if DRY_RUN else ‘LIVE TRADING’}\n”
f”Regime: {re_emoji} {regime.upper()}\n”
f”Take Profit: {TAKE_PROFIT_PCT}% | Stop Loss: {STOP_LOSS_PCT}%\n”
f”Bull: TP {BULL_TAKE_PROFIT}% / SL {BULL_STOP_LOSS}%\n”
f”Bear: TP {BEAR_TAKE_PROFIT}% / SL {BEAR_STOP_LOSS}%\n”
f”Base Buy: ${params[‘current_buy_amount’]:.2f} (max ${MAX_BUY_AMOUNT})\n”
f”Compound: {int(COMPOUND_RATE*100)}% of profits\n”
f”Summary at {SUMMARY_TIME}”
)
if INTERVAL == “hourly”:
schedule.every().hour.do(run_dca)
elif INTERVAL == “daily”:
schedule.every().day.at(“09:00”).do(run_dca)
elif INTERVAL == “weekly”:
schedule.every().monday.at(“09:00”).do(run_dca)
schedule.every().day.at(SUMMARY_TIME).do(send_daily_summary)
# Schedule opportunity buy check every 30 minutes after 12 PM
schedule.every(30).minutes.do(check_opportunity_buy)

```
print(f"\nScheduler running. Buys at 09:00, summary at {SUMMARY_TIME}.\n")
print(f"Opportunity buy: checks every 30min after 12PM (hourly RSI < {OPPORTUNITY_RSI_THRESHOLD})\n")
while True:
    schedule.run_pending()
    check_telegram_commands()
    time.sleep(30)
```

if **name** == “**main**”:
start()