import sys
import time
import json
import datetime
import requests
import pandas as pd
import MetaTrader5 as mt5

# --- STRATEGY & INSTRUMENT CONFIGURATION ---
SYMBOL = "XAUUSDm"
TIMEFRAME = mt5.TIMEFRAME_M5      # 5-Minute Execution Timeframe
LOOKBACK_BARS = 60                # Historical M5 bars for indicators
MAGIC_NUMBER = 202611

# --- OLLAMA AI CONFIGURATION ---
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"

# --- RISK MANAGEMENT & FILTER PARAMETERS ---
DEFAULT_LOT_SIZE = 0.01           # Base lot size
MAX_OPEN_POSITIONS = 5           # Max concurrent active positions
MAX_RISK_PERCENT = 0.03          # Max account risk cap (3%)
MAX_SL_DIST_DOLLARS = 4.00       # Maximum SL cap ($4.00)
MIN_SL_DIST_DOLLARS = 1.50       # Minimum SL floor ($1.50)
MAX_ALLOWED_SPREAD = 0.45        # Hard cap spread tolerance ($0.45)
MIN_M5_ATR_DOLLARS = 1.00        # Minimum ATR to avoid low-volatility chop
ATR_SL_MULTIPLIER = 2.0          # Dynamic ATR SL multiplier
RISK_REWARD_RATIO = 1.25         # Target Risk-to-Reward ratio (1:1.25)
MAX_CONSECUTIVE_LOSSES = 4       # Consecutive loss safety lock

# --- SESSION TIMING (UTC) ---
SESSION_START_HOUR_UTC = 7       # London Open (07:00 UTC)
SESSION_END_HOUR_UTC = 20        # NY Session Mid-Close (20:00 UTC)


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


def is_active_trading_session() -> bool:
    """Restricts trading to London and New York high-liquidity sessions (07:00-20:00 UTC)."""
    now_utc = datetime.datetime.now(datetime.timezone.utc).time()
    start_time = datetime.time(SESSION_START_HOUR_UTC, 0)
    end_time = datetime.time(SESSION_END_HOUR_UTC, 0)
    return start_time <= now_utc <= end_time


def count_open_positions(symbol: str, magic: int) -> int:
    """Counts active positions for the given symbol and magic number."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return 0
    return sum(1 for pos in positions if pos.magic == magic)


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


def get_h1_htf_bias(symbol: str) -> str:
    """Evaluates H1 Higher Timeframe trend via EMA 50 / 200 crossover."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 200)
    if rates is None or len(rates) < 200:
        print("[HTF WARNING] Could not fetch sufficient H1 candles. Allowing neutral bias.")
        return "ANY"

    df_h1 = pd.DataFrame(rates)
    ema_50 = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[-1]
    ema_200 = df_h1['close'].ewm(span=200, adjust=False).mean().iloc[-1]

    if ema_50 > ema_200:
        return "BUY_ONLY"
    elif ema_50 < ema_200:
        return "SELL_ONLY"
    return "ANY"


def calculate_m5_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates smoothed M5 trend indicators (EMA 8/21, RSI-14, ATR-14)."""
    df['ema_fast'] = df['close'].ewm(span=8, adjust=False).mean()
    df['ema_slow'] = df['close'].ewm(span=21, adjust=False).mean()

    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

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


def ask_ollama_model(df: pd.DataFrame, htf_bias: str) -> str:
    """Forces Ollama to evaluate structured M5 momentum constrained by HTF bias."""
    latest = df.iloc[-1]
    prev = df.iloc[-2]

    fast_above = latest['ema_fast'] > latest['ema_slow']
    rsi = latest['rsi']
    price_change = latest['close'] - prev['close']

    # Baseline Python decision logic
    if htf_bias == "BUY_ONLY":
        fallback_bias = "BUY"
    elif htf_bias == "SELL_ONLY":
        fallback_bias = "SELL"
    else:
        fallback_bias = "BUY" if (fast_above or rsi >= 50 or price_change > 0) else "SELL"

    prompt = f"""
    You are an M5 Gold (XAUUSD) Trend Scalper.
    You MUST output either "BUY" or "SELL". HOLD IS NOT ALLOWED.

    H1 Trend Context: {htf_bias}
    M5 Market Indicators:
    - EMA(8): {latest['ema_fast']:.2f}
    - EMA(21): {latest['ema_slow']:.2f}
    - RSI(14): {rsi:.1f}
    - ATR(14): {latest['atr']:.2f}
    - Calculated Bias: {fallback_bias}

    Execution Constraint:
    - If H1 Context is BUY_ONLY, choose "BUY" unless momentum is severely overbought.
    - If H1 Context is SELL_ONLY, choose "SELL" unless momentum is severely oversold.

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

            # Enforce hard filter alignment with HTF
            if htf_bias == "BUY_ONLY" and decision == "SELL":
                print(f"   [HTF FILTER] Overriding SELL signal -> Forced BUY due to Bullish H1 Trend.")
                return "BUY"
            elif htf_bias == "SELL_ONLY" and decision == "BUY":
                print(f"   [HTF FILTER] Overriding BUY signal -> Forced SELL due to Bearish H1 Trend.")
                return "SELL"

            if decision in ["BUY", "SELL"]:
                return decision

        print(f"   [AI Fallback] Defaulting to baseline bias: {fallback_bias}")
        return fallback_bias
    except Exception as e:
        print(f"   [OLLAMA ERROR] Signal request failed: {e}. Defaulting to: {fallback_bias}")
        return fallback_bias


def execute_order(symbol: str, action: str, lot_size: float = DEFAULT_LOT_SIZE) -> str:
    """Executes market orders with structural safety and volatility filters."""
    if action.upper() not in ["BUY", "SELL"]:
        return f"Execution error: Invalid signal '{action}'."

    if not init_mt5():
        return "Trade aborted: MT5 Initialization failed."

    # Guardrail 1: Active Open Positions Limit Check
    open_positions = count_open_positions(symbol, MAGIC_NUMBER)
    if open_positions >= MAX_OPEN_POSITIONS:
        mt5.shutdown()
        return f"[POSITION GUARD] Skipped: Active positions ({open_positions}) reached max limit ({MAX_OPEN_POSITIONS})."

    # Guardrail 2: Loss Streak Check
    if check_consecutive_losses(symbol, MAX_CONSECUTIVE_LOSSES):
        mt5.shutdown()
        return f"[RISK GUARD] Blocked: Hit {MAX_CONSECUTIVE_LOSSES} consecutive losses."

    # Guardrail 3: Session Time Check
    if not is_active_trading_session():
        mt5.shutdown()
        return "[TIME GUARD] Skipped: Current time outside active London/NY sessions (07:00-20:00 UTC)."

    account_info = mt5.account_info()
    symbol_info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)

    if not account_info or not symbol_info or not tick:
        mt5.shutdown()
        return "Account or Symbol market context unavailable."

    balance = account_info.balance
    max_allowed_loss = balance * MAX_RISK_PERCENT

    # Guardrail 4: Spread Cap
    spread = tick.ask - tick.bid
    if spread > MAX_ALLOWED_SPREAD:
        mt5.shutdown()
        return f"[SPREAD GUARD] Rejected: Current spread (${spread:.2f}) exceeds max allowed limit (${MAX_ALLOWED_SPREAD:.2f})."

    # Fetch M5 ATR for Volatility Verification
    df = get_market_data(symbol, count=20)
    if df is None or df['atr'].dropna().empty:
        mt5.shutdown()
        return "Trade aborted: Volatility indicators unavailable."

    latest_atr = df['atr'].iloc[-1]

    # Guardrail 5: Low-Volatility Chop & Relative Spread Guard
    if latest_atr < MIN_M5_ATR_DOLLARS:
        mt5.shutdown()
        return f"[VOLATILITY GUARD] Skipped: Low ATR (${latest_atr:.2f}). Market in range chop."

    if spread > (latest_atr * 0.15):
        mt5.shutdown()
        return f"[SPREAD GUARD] Rejected: Spread (${spread:.2f}) exceeds 15% of current M5 ATR (${latest_atr * 0.15:.2f})."

    # Dynamic SL Calculation
    raw_sl_distance = latest_atr * ATR_SL_MULTIPLIER
    sl_distance = min(max(raw_sl_distance, MIN_SL_DIST_DOLLARS), MAX_SL_DIST_DOLLARS)

    projected_loss = sl_distance * (lot_size / 0.01)
    if projected_loss > max_allowed_loss:
        mt5.shutdown()
        return f"[RISK GUARD] Rejected: Risk (${projected_loss:.2f}) exceeds limit (${max_allowed_loss:.2f})."

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
    print(f"Starting Continuous Structural M5 Gold Scalper ({SYMBOL}). Press Ctrl+C to exit.\n")

    while True:
        try:
            if init_mt5():
                htf_bias = get_h1_htf_bias(SYMBOL)
                df_candles = get_market_data(SYMBOL, LOOKBACK_BARS)
                mt5.shutdown()

                if df_candles is not None:
                    current_time = df_candles.iloc[-1]['time']
                    print(f"\n[{current_time}] Analyzing Market State | H1 Bias: {htf_bias}")

                    decision = ask_ollama_model(df_candles, htf_bias)
                    print(f"   [AI Decision] {decision}")

                    print(f"   Executing {decision} order sequence on MT5...")
                    status = execute_order(SYMBOL, decision, lot_size=DEFAULT_LOT_SIZE)
                    print(f"   [Execution Result] {status}")

            time.sleep(300)  # Sleep 5 minutes for next M5 cycle

        except KeyboardInterrupt:
            print("\nBot execution halted by user.")
            sys.exit(0)
        except Exception as e:
            print(f"Unexpected Loop Error: {e}")
            time.sleep(10)