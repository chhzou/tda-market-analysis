import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import StandardScaler

def fetch_data(ticker="^GSPC", start="2019-01-01", end="2021-01-01"):
    """
    Fetches daily closing prices from Yahoo Finance.
    Defaults to S&P 500 during the COVID-19 crash period.
    """
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, start=start, end=end, progress=False)
    
    # Handle multi-level columns if they exist (yfinance update quirk)
    if isinstance(data.columns, pd.MultiIndex):
        data = data['Close']
    else:
        data = data['Close']
        
    return data

def get_log_returns(prices):
    """
    Computes log returns to stationarize the time series.
    """
    # prices is a DataFrame/Series. We take values, ensuring 1D array.
    p = prices.values.flatten()
    # Log returns: ln(P_t / P_{t-1})
    return np.diff(np.log(p + 1e-9)) # small epsilon to avoid log(0)

def takens_embedding(time_series, dim=3, delay=1):
    """
    Creates a sliding window embedding (Takens' Embedding).
    Transforms a 1D time series into a 'dim'-dimensional point cloud.
    """
    n = len(time_series)
    if n < dim * delay:
        raise ValueError("Time series too short for embedding.")
    
    # Create the matrix of sliding windows
    # Shape: (n_windows, dim)
    embedded = np.array([time_series[i : i + dim * delay : delay] 
                         for i in range(n - (dim * delay) + 1)])
    return embedded

def main():
    # 1. Load Data
    prices = fetch_data()
    
    if prices.empty:
        print("Error: No data fetched. Check your internet connection.")
        return

    # 2. Preprocessing
    # We focus on a specific volatile window to see the topology clearly
    # Let's look at the crash (Feb 2020 - April 2020)
    focus_start = "2020-02-01"
    focus_end = "2020-05-01"
    
    # Filter by date index
    mask = (prices.index >= focus_start) & (prices.index <= focus_end)
    crash_prices = prices.loc[mask]
    
    # Compute log returns (volatility proxy)
    log_ret = get_log_returns(crash_prices)
    
    # 3. Embedding (The "Geometry" Step)
    # We embed the 1D returns into a 20-dimensional space to find "shapes"
    window_size = 20
    point_cloud = takens_embedding(log_ret, dim=window_size, delay=1)
    
    # Normalize point cloud (standard practice for TDA)
    scaler = StandardScaler()
    point_cloud_scaled = scaler.fit_transform(point_cloud)
    
    # 4. Persistent Homology (The "Topology" Step)
    print("Computing Persistence Diagrams (this may take a moment)...")
    
    # ripser computes H0 (connected components) and H1 (loops)
    # maxdim=1 means we look for 1D loops (volatility cycles)
    result = ripser(point_cloud_scaled, maxdim=1)
    diagrams = result['dgms']
    
    # 5. Visualization
    fig = plt.figure(figsize=(14, 6))
    
    # Plot A: The Time Series
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(crash_prices.index, crash_prices.values, color='black', label='S&P 500')
    ax1.set_title(f"Market Crash: {focus_start} to {focus_end}")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # Plot B: The Persistence Diagram
    ax2 = fig.add_subplot(1, 2, 2)
    plot_diagrams(diagrams, show=False, ax=ax2)
    ax2.set_title("Persistence Diagram (H0 and H1)")
    
    plt.tight_layout()
    plt.savefig("tda_result.png")
    print("Analysis Complete. Plot saved to 'tda_result.png'.")
    plt.show()

if __name__ == "__main__":
    main()
