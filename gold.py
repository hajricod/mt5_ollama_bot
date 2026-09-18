import sys
import time
import json
import requests
import pandas as pd
import MetaTrader5 as mt5

# --- STRATEGY & INSTRUMENT CONFIGURATION ---
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5      # Reverted to 5-Minute Timeframe
LOOKBACK_BARS = 60                # Historical M5 bars for indicators
MAGIC_NUMBER = 202611

# --- OLLAMA AI CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

# --- REFINED M5 RISK MANAGEMENT PARAMETERS ---
DEFAULT_LOT_SIZE = 0.01          # Base lot size
MAX_RISK_PERCENT = 0.03          # Risk capped at 3% per trade
MAX_SL_DIST_DOLLARS = 4.00       # Maximum Stop Loss cap ($4.00 on Gold)
MIN_SL_DIST_DOLLARS = 1.50       # Minimum Stop Loss floor ($1.50)
MAX_ALLOWED_SPREAD = 0.45        # Max spread tolerance ($0.45)
ATR_SL_MULTIPLIER = 2.0          # SL multiplier (2.0x ATR for volatility buffer)
RISK_REWARD_RATIO = 1.25         # TP multiplier (1:1.25 Risk-to-Reward)
MAX_CONSECUTIVE_LOSSES = 100     # Loss streak safety limit


def init_mt5() -> bool:
    """Initializes MetaTrader 5 connection and selects the symbol."""
    if not mt5.initialize():
        print(f"[MT5 ERROR] Initialization failed: {mt5.last_error()}")
        return False

    if not mt5.symbol_select(SYMBOL, True):
        print(f"[MT5 ERROR] Symbol '{SYMBOL}' not found or inactive in Market Watch.")
        mt5.shutdown()
        return False

    return True


def get_broker_filling_type(symbol_info) -> int:
    """Determines valid execution order filling mode dynamically."""
    filling = symbol_info.filling_mode
    if filling & 1:
        return mt5.ORDER_FILLING_FOK
    elif filling & 2:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def normalize_lot(symbol_info, target_lot: float) -> float:
    """Clamps requested lot size within broker specifications."""
    step = symbol_info.volume_step
    min_vol = symbol_info.volume_min
    max_vol = symbol_info.volume_max

    lot = round(target_lot / step) * step
    return max(min_vol, min(lot, max_vol))


def check_consecutive_losses(symbol: str, limit: int = MAX_CONSECUTIVE_LOSSES) -> bool:
    """Checks trade history for recent consecutive losing trades."""
    now = time.time()
    history = mt5.history_deals_get(now - (86400 * 2), now)
    if history is None or len(history) == 0:
        return False

    losses = 0
    for deal in reversed(history):
        if deal.symbol == symbol and deal.entry == mt5.DEAL_ENTRY_OUT:
            if deal.profit < 0:
                losses += 1
                if losses >= limit:
                    return True
            elif deal.profit > 0:
                break
    return False


def calculate_m5_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates smoothed trend indicators (EMA 8/21, RSI-14, ATR-14) for M5 candles."""
    df['ema_fast'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()

    # Standard RSI (14-period)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # Standard Average True Range (14-period)
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(window=14).mean()

    return df


def get_market_data(symbol: str, count: int = LOOKBACK_BARS) -> pd.DataFrame | None:
    """Fetches M5 historical rates and calculates indicators."""
    rates = mt5.copy_rates_from_pos(symbol, TIMEFRAME, 0, count)
    if rates is None or len(rates) == 0:
        print(f"[MT5 ERROR] Failed to fetch M5 market data for {symbol}.")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df = calculate_m5_indicators(df)

    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume', 'ema_fast', 'ema_slow', 'rsi', 'atr']]


def ask_ollama_model(df: pd.DataFrame) -> str:
    """Forces Ollama to evaluate structured M5 momentum and choose BUY or SELL without HOLD."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    fast_above = latest['ema_fast'] > latest['ema_slow']
    rsi = latest['rsi']
    price_change = latest['close'] - prev['close']

    # Determine baseline bias in Python
    if fast_above or rsi >= 50 or price_change > 0:
        fallback_bias = "BUY"
    else:
        fallback_bias = "SELL"

    prompt = f"""
    You are an M5 Gold (XAUUSD) Trend-Following Scalper.
    You MUST execute a trade on every cycle. HOLD IS NOT ALLOWED.

    M5 Market Indicators:
    - EMA(8): {latest['ema_fast']:.2f}
    - EMA(21): {latest['ema_slow']:.2f}
    - RSI(14): {rsi:.1f}
    - ATR(14): {latest['atr']:.2f}
    - Technical Trend Bias: {fallback_bias}

    Execution Rules:
    - Choose "BUY" if EMA(8) > EMA(21) or RSI >= 50
    - Choose "SELL" if EMA(8) < EMA(21) or RSI < 50

    Respond STRICTLY in JSON format:
    {{"reason": "short explanation", "decision": "BUY" | "SELL"}}
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0.1,
            "num_predict": 40
        }
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=20)
        if response.status_code == 200:
            res_json = json.loads(response.json().get('response', '{}'))
            reason = res_json.get('reason', '')
            decision = res_json.get('decision', '').strip().upper()

            print(f"   [AI Reasoning] {reason}")
            if decision in ["BUY", "SELL"]:
                return decision

        print(f"   [AI Fallback] Defaulting to baseline bias: {fallback_bias}")
        return fallback_bias
    except Exception as e:
        print(f"   [OLLAMA ERROR] Signal request failed: {e}. Defaulting to: {fallback_bias}")
        return fallback_bias


def execute_order(symbol: str, action: str, lot_size: float = DEFAULT_LOT_SIZE) -> str:
    """Executes market orders with M5 ATR risk parameters."""
    if action.upper() not in ["BUY", "SELL"]:
        return f"Execution error: Invalid signal '{action}'."

    if not init_mt5():
        return "Trade aborted: MT5 Initialization failed."

    # Guardrail 1: Loss Streak Check
    if check_consecutive_losses(symbol, MAX_CONSECUTIVE_LOSSES):
        mt5.shutdown()
        return f"[RISK GUARD] Blocked: Hit {MAX_CONSECUTIVE_LOSSES} consecutive losses."

    # Guardrail 2: Account Budget Verification
    account_info = mt5.account_info()
    if not account_info:
        mt5.shutdown()
        return "Failed to fetch account balance."

    balance = account_info.balance
    max_allowed_loss = balance * MAX_RISK_PERCENT

    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)

    if not symbol_info or not tick:
        mt5.shutdown()
        return "Symbol or tick data unavailable."

    # Guardrail 3: Spread Guardrail
    spread = tick.ask - tick.bid
    if spread > MAX_ALLOWED_SPREAD:
        mt5.shutdown()
        return f"[SPREAD GUARD] Rejected: Current spread (${spread:.2f}) exceeds limit (${MAX_ALLOWED_SPREAD:.2f})."

    # Fetch ATR data
    df = get_market_data(symbol, count=20)
    if df is None or df['atr'].dropna().empty:
        mt5.shutdown()
        return "Trade aborted: Indicator data unavailable."

    latest_atr = df['atr'].iloc[-1]
    raw_sl_distance = latest_atr * ATR_SL_MULTIPLIER

    # Bound SL distance for M5 market noise
    sl_distance = min(max(raw_sl_distance, MIN_SL_DIST_DOLLARS), MAX_SL_DIST_DOLLARS)

    projected_loss = sl_distance * (lot_size / 0.01)
    if projected_loss > max_allowed_loss:
        mt5.shutdown()
        return f"[RISK GUARD] Rejected: Risk (${projected_loss:.2f}) exceeds allowed limit (${max_allowed_loss:.2f})."

    tp_distance = sl_distance * RISK_REWARD_RATIO
    digits = symbol_info.digits
    filling_type = get_broker_filling_type(symbol_info)
    lot_size = normalize_lot(symbol_info, lot_size)

    if action.upper() == "BUY":
        trade_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - sl_distance
        tp = price + tp_distance
    else:  # SELL
        trade_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + sl_distance
        tp = price - tp_distance

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot_size),
        "type": trade_type,
        "price": price,
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "deviation": 25,
        "magic": MAGIC_NUMBER,
        "comment": f"M5 Scalp {action}",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": filling_type,
    }

    result = mt5.order_send(request)
    mt5.shutdown()

    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        comment = result.comment if result else "Null MT5 response"
        return f"Order Failed: {comment} (Code: {result.retcode if result else 'N/A'})"

    return (f"Order Executed! Ticket: {result.order} | Price: {result.price} | "
            f"SL: {round(sl, digits)} (-${sl_distance:.2f}) | TP: {round(tp, digits)} (+${tp_distance:.2f})")


# --- CONTINUOUS 5-MINUTE AUTOMATED LOOP ---
if __name__ == "__main__":
    print(f"Starting Continuous M5 Trend Scalper ({SYMBOL}). Press Ctrl+C to exit.\n")

    while True:
        try:
            if init_mt5():
                df_candles = get_market_data(SYMBOL, LOOKBACK_BARS)
                mt5.shutdown()

                if df_candles is not None:
                    current_time = df_candles.iloc[-1]['time']
                    print(f"\n[{current_time}] Analyzing M5 Candle Data...")

                    decision = ask_ollama_model(df_candles)
                    print(f"   [AI Decision] {decision}")

                    print(f"   Executing {decision} order on MT5...")
                    status = execute_order(SYMBOL, decision, lot_size=DEFAULT_LOT_SIZE)
                    print(f"   [Execution Result] {status}")

            time.sleep(300)  # Sleep 5 minutes for next M5 cycle

        except KeyboardInterrupt:
            print("\nBot execution halted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"Unexpected Loop Error: {e}")
            time.sleep(10)