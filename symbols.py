import MetaTrader5 as mt5

if mt5.initialize():
    symbols = mt5.symbols_get()
    print("Found", len(symbols), "total symbols.")
    
    # Filter and print all symbols matching EURUSD
    eur_symbols = [s.name for s in symbols if "EURUSD" in s.name.upper()]
    print("Available EURUSD variants:", eur_symbols)
    
    mt5.shutdown()
else:
    print("Failed to connect to MT5")