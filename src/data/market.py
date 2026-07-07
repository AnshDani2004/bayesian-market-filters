import requests
import pandas as pd
import numpy as np
import time

class BinanceDataClient:
    """
    Client to fetch free historical OHLCV data from Binance REST API.
    """
    BASE_URL = "https://api.binance.com/api/v3/klines"
    
    @classmethod
    def fetch_data(cls, symbol: str = "BTCUSDT", interval: str = "1m", limit: int = 1000, max_points: int = 2000) -> pd.DataFrame:
        """
        Fetches historical klines (candlesticks).
        Paginates backwards from the current time if max_points > 1000.
        """
        all_klines = []
        end_time = None
        
        while len(all_klines) < max_points:
            params = {
                "symbol": symbol,
                "interval": interval,
                "limit": min(1000, max_points - len(all_klines))
            }
            if end_time:
                params["endTime"] = end_time
                
            response = requests.get(cls.BASE_URL, params=params)
            
            try:
                response.raise_for_status()
            except requests.exceptions.HTTPError as e:
                if response.status_code == 451:
                    print("Binance API blocked (451 Unavailable For Legal Reasons). Falling back to yfinance.")
                    import yfinance as yf
                    yf_symbol = symbol.replace("USDT", "-USD")
                    
                    if interval == "1h":
                        period = "730d"
                    elif interval == "1m":
                        period = "7d"
                    else:
                        period = "max"
                        
                    df = yf.download(yf_symbol, interval=interval, period=period, progress=False)
                    df.reset_index(inplace=True)
                    df.rename(columns={'Datetime': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                    # For yfinance multi-index columns, sometimes it returns tuples
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                        df.rename(columns={'Datetime': 'timestamp', 'Open': 'open', 'High': 'high', 'Low': 'low', 'Close': 'close', 'Volume': 'volume'}, inplace=True)
                    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].dropna()
                else:
                    raise e
            
            data = response.json()
            if not data:
                break
                
            all_klines = data + all_klines # Prepend older data
            end_time = data[0][0] - 1      # Set new end_time to just before the oldest fetched candle
            
            # Simple rate limiting protection
            time.sleep(0.1)
            
        columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time', 
                   'quote_asset_volume', 'number_of_trades', 'taker_buy_base', 'taker_buy_quote', 'ignore']
        df = pd.DataFrame(all_klines, columns=columns)
        
        # Keep only what we need, convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        return df


class Backtester:
    """
    Event-driven simulation engine tracking PnL with fee deduction.
    """
    def __init__(self, taker_fee_bps: float = 5.0):
        self.taker_fee = taker_fee_bps / 10000.0
        
    def run(self, df: pd.DataFrame, signals: pd.Series) -> pd.DataFrame:
        """
        Runs the backtest using the provided price series and target position signals.
        
        Args:
            df: DataFrame containing 'close' prices.
            signals: Series containing target positions (-1.0 to 1.0)
            
        Returns:
            DataFrame with appended PnL metrics.
        """
        df = df.copy()
        df['target_position'] = signals
        df['current_position'] = df['target_position'].shift(1).fillna(0) # Position executed at next open/close
        
        # Calculate returns
        # Strategy return = Position * Asset Return
        df['asset_return'] = df['close'].pct_change()
        df['gross_return'] = df['current_position'] * df['asset_return']
        
        # Calculate fee deduction
        # We pay fees whenever our position changes
        df['position_change'] = df['current_position'].diff().fillna(0).abs()
        df['fee_drag'] = df['position_change'] * self.taker_fee
        
        df['net_return'] = df['gross_return'] - df['fee_drag']
        
        # Cumulative PnL
        df['cum_gross'] = (1 + df['gross_return'].fillna(0)).cumprod() - 1
        df['cum_net'] = (1 + df['net_return'].fillna(0)).cumprod() - 1
        
        return df
        
    @staticmethod
    def calculate_metrics(df: pd.DataFrame, periods_per_year: int = 525600) -> dict:
        """
        Calculates performance metrics.
        Assumes 1-minute data by default (525,600 mins per year).
        """
        net_returns = df['net_return'].dropna()
        
        # Annualized Sharpe (assuming risk-free rate = 0)
        mean_return = net_returns.mean()
        std_return = net_returns.std()
        
        if std_return == 0:
            sharpe = 0.0
        else:
            sharpe = (mean_return / std_return) * np.sqrt(periods_per_year)
            
        # Max Drawdown
        cum_net = (1 + net_returns).cumprod()
        rolling_max = cum_net.cummax()
        drawdowns = (cum_net - rolling_max) / rolling_max
        max_dd = drawdowns.min()
        
        # Hit Rate (Win rate of non-zero return periods)
        active_returns = net_returns[df['current_position'] != 0]
        if len(active_returns) > 0:
            hit_rate = len(active_returns[active_returns > 0]) / len(active_returns)
        else:
            hit_rate = 0.0
            
        return {
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'hit_rate': hit_rate,
            'total_net_pnl': cum_net.iloc[-1] - 1 if len(cum_net) > 0 else 0.0
        }
