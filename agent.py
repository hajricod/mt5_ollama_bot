import json
import MetaTrader5 as mt5
import pandas as pd
from ollama import chat, ChatResponse

# Change this if your broker uses a suffix (e.g., "EURUSDm", "EURUSD.a")
SYMBOL = "EURUSDm"

def fetch_recent_candles(symbol: str, count: int = 10) -> str:
    """Fetches OHLC candles from MT5 after ensuring the symbol is enabled."""
    if not mt5.initialize():
        return f"MT5 Init Error: {mt5.last_error()}"

    # Ensure symbol is visible in Market Watch
    if not mt5.symbol_select(symbol, True):
        mt5.shutdown()
        return f"Symbol '{symbol}' not found or disabled in Market Watch. Check if your broker uses a suffix."

    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, count)
    mt5.shutdown()

    if rates is None or len(rates) == 0:
        return f"No market data received for {symbol}. Try opening MT5 and opening a chart for {symbol}."

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df[['time', 'open', 'high', 'low', 'close', 'tick_volume']].to_string(index=False)


def execute_order(symbol: str, action: str, lot_size: float = 0.01, sl_points: int = 100, tp_points: int = 200) -> str:
    """Executes a market order on MT5."""
    if not mt5.initialize():
        return f"MT5 Init Error: {mt5.last_error()}"

    mt5.symbol_select(symbol, True)
    symbol_info = mt5.symbol_info(symbol)
    if not symbol_info:
        mt5.shutdown()
        return f"Symbol '{symbol}' not found."

    point = symbol_info.point
    tick = mt5.symbol_info_tick(symbol)

    if action.upper() == "BUY":
        trade_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - (sl_points * point)
        tp = price + (tp_points * point)
    elif action.upper() == "SELL":
        trade_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + (sl_points * point)
        tp = price - (tp_points * point)
    else:
        mt5.shutdown()
        return "Invalid action. Choose BUY or SELL."

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": float(lot_size),
        "type": trade_type,
        "price": price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": 100200,
        "comment": "Ollama Trading Bot",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    mt5.shutdown()

    if result.retcode != mt5.TRADE_RETCODE_DONE:
        return f"Trade Failed: {result.comment} (Code: {result.retcode})"

    return f"Trade Success! Order ID: {result.order} at Price: {result.price}"


def run_bot():
    print(f"Fetching market data for {SYMBOL}...")
    candles = fetch_recent_candles(SYMBOL)
    print("\n--- Recent Market Data ---\n", candles)

    # Stop execution if market data was not fetched
    if "Error" in candles or "No market data" in candles:
        print("\nAborting cycle due to missing market data.")
        return

    system_prompt = (
        "You are an automated Forex trading assistant. "
        "Analyze the provided OHLC market data. "
        "If there is a clear trend, call `execute_order` with action BUY or SELL and lot_size 0.01. "
        "If price movement is ranging or uncertain, do NOT call any tool and explain why."
    )

    print("\nAnalyzing market with qwen2.5:3b (CPU Mode)...")
    try:
        response: ChatResponse = chat(
            model='qwen2.5:3b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': f"Symbol: {SYMBOL}\nData:\n{candles}"}
            ],
            tools=[execute_order],
            options={"num_gpu": 0}  # Forces CPU execution to prevent GPU CUDA crashes
        )

        if response.message.tool_calls:
            for call in response.message.tool_calls:
                func_name = call.function.name
                args = call.function.arguments
                print(f"\n[AI Action Triggered]: {func_name}({args})")

                if func_name == "execute_order":
                    result = execute_order(**args)
                    print(f"[Execution Status]: {result}")
        else:
            print("\n[AI Analysis - No Order Executed]:")
            print(response.message.content)

    except Exception as e:
        print(f"\n[Ollama Connection Error]: {e}")


if __name__ == "__main__":
    run_bot()