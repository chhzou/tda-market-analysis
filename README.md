# Topological Data Analysis (TDA) for Financial Regime Shift Detection

## Project Overview
This project applies **Algebraic Topology** to financial time-series analysis. By treating market returns as a high-dimensional manifold, we use **Persistent Homology** to detect structural changes (regime shifts) that traditional statistical methods (like moving averages) might miss.

## Key Mathematical Concepts
* **Takens' Embedding Theorem:** Reconstructs the phase space of a dynamical system from a single observation variable (stock returns).
* **Vietoris-Rips Filtration:** A method to build a simplicial complex from the point cloud data.
* **Persistent Homology:** Computes the birth and death of topological features ($H_0$ components and $H_1$ loops) across different spatial scales. In finance, high persistence in $H_1$ often correlates with **market instability**.

## Technical Stack
* **TDA Engine:** `ripser` (C++ optimized computation of cohomology)
* **Visualization:** `persim`, `matplotlib`
* **Data Processing:** `pandas`, `numpy`, `scikit-learn`

## Usage
1.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
2.  Run the analysis:
    ```bash
    python analysis.py
    ```
3.  Output:
    * Generates `tda_result.png` showing the S&P 500 crash period alongside its persistence diagram.

## Results Interpretation
* **Points far from the diagonal** in the Persistence Diagram represent robust topological features (significant loops/cycles in the market state).
* **Points near the diagonal** represent topological noise.
* During the 2020 crash, the topology of the return manifold shifts significantly, creating distinct high-persistence features.
