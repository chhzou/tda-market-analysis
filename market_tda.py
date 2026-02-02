import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from ripser import ripser
from persim import plot_diagrams
from sklearn.preprocessing import StandardScaler

# 1. Get Data (S&P 500 during 2008 or 2020)
print("Fetching market data...")
data = yf.download("^GSPC", start="2019-01-01", end="2021-01-01")['Close']
returns = data.pct_change().dropna().values.reshape(-1, 1)

# 2. Sliding Window Embedding (Takens' Embedding)
# We turn a 1D time series into a high-dimensional point cloud
def create_point_cloud(series, window_size=20):
    return np.array([series[i:i+window_size].flatten() for i in range(len(series)-window_size)])

point_cloud = create_point_cloud(returns, window_size=10)
scaler = StandardScaler()
point_cloud_scaled = scaler.fit_transform(point_cloud)

# 3. Compute Persistent Homology (The "Math" Part)
# We only use a subset of points for the demo speed
print("Computing Persistence Diagrams...")
result = ripser(point_cloud_scaled[:500], maxdim=1)
diagrams = result['dgms']

# 4. Visualization
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(data.values)
plt.title("S&P 500 Price (2020 Crash Period)")

plt.subplot(1, 2, 2)
plot_diagrams(diagrams, show=True)
# H0 (dots) = Connected components
# H1 (circles) = Topological cycles/holes in the data manifold
