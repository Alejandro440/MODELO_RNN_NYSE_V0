import pandas as pd
import yfinance as yf

from pipeline_config import INDEX_PROCESSED_DIR, RAW_INDICES_DIR, ensure_data_directories


def download_sp500_index(start_date="2000-01-01"):
    """Descarga el índice S&P 500 desde Yahoo Finance y lo guarda en crudo."""

    data = yf.download("^GSPC", start=start_date, progress=False)
    data.reset_index(inplace=True)

    raw_output = RAW_INDICES_DIR / "sp500.csv"
    data.to_csv(raw_output, index=False)

    return data


def load_and_clean_data(dataframe):
    data = dataframe.copy()
    data['Date'] = pd.to_datetime(data['Date'])
    data.sort_values('Date', inplace=True)
    data.drop_duplicates(inplace=True)
    data.dropna(inplace=True)
    return data

def calculate_sma(data, window):
    return data.rolling(window=window).mean()

def calculate_ema(prices, days):
    return prices.ewm(span=days, adjust=False).mean()

def calculate_macd(data):
    ema12 = calculate_ema(data['Close'], 12)
    ema26 = calculate_ema(data['Close'], 26)
    macd = ema12 - ema26
    signal_line = calculate_ema(macd, 9)
    return macd, signal_line

def calculate_rsi(data, window=14):
    change = data['Close'].diff()
    gain = (change.where(change > 0, 0)).rolling(window=window).mean()
    loss = (-change.where(change < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def add_technical_indicators(data):
    data['SMA_30'] = calculate_sma(data['Close'], 30)
    data['SMA_50'] = calculate_sma(data['Close'], 50)
    data['EMA_30'] = calculate_ema(data['Close'], 30)
    data['EMA_50'] = calculate_ema(data['Close'], 50)
    data['MACD'], data['Signal_Line'] = calculate_macd(data)
    data['RSI'] = calculate_rsi(data)
    return data


def process_index(start_date="2000-01-01"):
    data = download_sp500_index(start_date)
    data = load_and_clean_data(data)
    data = add_technical_indicators(data)
    data.dropna(inplace=True)

    output_filepath = INDEX_PROCESSED_DIR / "sp500_processed.csv"
    data.to_csv(output_filepath, index=False)
    print(f'Índice S&P 500 procesado y guardado en {output_filepath}')


def main():
    ensure_data_directories()
    process_index()


if __name__ == "__main__":
    main()