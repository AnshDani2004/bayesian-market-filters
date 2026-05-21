import nbformat as nbf

# ---------------------------------------------------------
# Notebook 1: Kalman Filter
# ---------------------------------------------------------
nb_kalman = nbf.v4.new_notebook()

md_intro = """# Latent Fair Value Estimation via Adaptive Kalman Filtering

## Market Microstructure Motivation

In modern electronic limit order books, the observed execution "price" is rarely the true *fair value* of an asset. Microstructure noise—driven by bid-ask bounce, transient order flow imbalances, and discrete tick sizes—causes the observed price to constantly oscillate around a latent (unobservable) fair value.

If we assume the latent fair value $\mu_t$ follows a random walk with drift, and the observed price $y_t$ is a noisy manifestation of $\mu_t$, we can formulate this as a Linear State-Space Model. A standard Exponentially Weighted Moving Average (EWMA) is slow to react to sudden regime shifts. By contrast, a Kalman Filter explicitly models the uncertainty (variance) of the fair value and can adaptively update its internal noise matrices via Expectation-Maximization (EM) techniques when market conditions change.
"""

md_math = r"""## Mathematical Derivation

Let our state vector be $x_t = [\mu_t, \delta_t]^T$, where $\mu_t$ is fair value and $\delta_t$ is the drift.

**1. Predict Step**
The state transitions according to a constant velocity kinematic model:
$$x_{t|t-1} = F x_{t-1|t-1} + w_t, \quad w_t \sim \mathcal{N}(0, Q)$$
$$P_{t|t-1} = F P_{t-1|t-1} F^T + Q$$
Where $F = \begin{bmatrix} 1 & dt \\ 0 & 1 \end{bmatrix}$.

**2. Update Step**
We observe the noisy price $y_t$:
$$y_t = H x_{t} + v_t, \quad v_t \sim \mathcal{N}(0, R)$$
Where $H = \begin{bmatrix} 1 & 0 \end{bmatrix}$.

The innovation (residual) is $v_t = y_t - H x_{t|t-1}$.
The innovation covariance is $S_t = H P_{t|t-1} H^T + R$.
The Kalman Gain determines how much we trust the observation vs. our prediction:
$$K_t = P_{t|t-1} H^T S_t^{-1}$$

We update our posterior state and covariance:
$$x_{t|t} = x_{t|t-1} + K_t v_t$$
$$P_{t|t} = (I - K_t H) P_{t|t-1}$$

### Adaptive Q and R Estimation
In an adaptive setting, we recursively update the observation noise $R$ and process noise $Q$ using a Robbins-Monro style online EM update with a forgetting factor $\alpha$:
$$R_{t} = (1 - \alpha) R_{t-1} + \alpha (v_t^2 - H P_{t|t-1} H^T)$$
$$Q_{t} = (1 - \alpha) Q_{t-1} + \alpha (K_t v_t v_t^T K_t^T)$$
"""

code_run = """import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from benchmark_kalman import run_benchmark

# Run the benchmark comparing Adaptive Kalman against EWMA and SMA
run_benchmark()
"""

nb_kalman['cells'] = [
    nbf.v4.new_markdown_cell(md_intro),
    nbf.v4.new_markdown_cell(md_math),
    nbf.v4.new_code_cell(code_run)
]


# ---------------------------------------------------------
# Notebook 2: HMM
# ---------------------------------------------------------
nb_hmm = nbf.v4.new_notebook()

md_intro_hmm = """# Unsupervised Regime Detection via Hidden Markov Models

## Market Microstructure Motivation

Volatility is entirely unobservable. While we can compute realized variance over a rolling window, this creates a lagging indicator. Markets typically transition between distinct macroeconomic or liquidity "regimes"—periods of relative calm and periods of violent structural breaks.

To build an optimal position-sizing signal, we need an unsupervised method to detect the *probability* of being in a volatile regime in real-time. A Hidden Markov Model (HMM) allows us to probabilistically classify these regimes purely from the log-returns without manual thresholds.
"""

md_math_hmm = r"""## Mathematical Derivation

Let $z_t \in \{0, 1\}$ be the hidden regime (0: Quiet, 1: Volatile). Let $y_t$ be the observed log-return.
Our emissions are Gaussian: $P(y_t | z_t = i) = \mathcal{N}(y_t; \mu_i, \sigma_i^2)$.

**1. Online Filtering (Forward Algorithm)**
To compute the real-time probability of the regime $P(z_t | y_{1:t})$, we recursively apply:
$$\hat{\alpha}_t(i) = \sum_j A(j, i) \alpha_{t-1}(j)$$
$$\alpha_t(i) \propto \hat{\alpha}_t(i) \cdot P(y_t | z_t=i)$$

**2. Offline Parameter Calibration (Baum-Welch EM)**
To fit the transition matrix $A$ and the emission parameters $(\mu_i, \sigma_i)$, we use the offline E-M loop.

*E-Step (Forward-Backward)*:
Computes the smoothed posterior $\gamma_t(i) = P(z_t=i | y_{1:T})$ and the joint state probability $\xi_t(i,j) = P(z_t=i, z_{t+1}=j | y_{1:T})$.

*M-Step*:
We maximize the expected log-likelihood by updating parameters:
$$A(i, j) = \frac{\sum_{t=1}^{T-1} \xi_t(i, j)}{\sum_{t=1}^{T-1} \gamma_t(i)}$$
$$\sigma_i^2 = \frac{\sum_{t=1}^T \gamma_t(i) (y_t - \mu_i)^2}{\sum_{t=1}^T \gamma_t(i)}$$
"""

code_run_hmm = """import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), '..')))
from benchmark_hmm import run_benchmark

# Train the HMM via Baum-Welch and plot the online posterior
run_benchmark()
"""

nb_hmm['cells'] = [
    nbf.v4.new_markdown_cell(md_intro_hmm),
    nbf.v4.new_markdown_cell(md_math_hmm),
    nbf.v4.new_code_cell(code_run_hmm)
]

with open('/Users/ansh/Proj_Res_3/notebook_01_kalman.ipynb', 'w') as f:
    nbf.write(nb_kalman, f)

with open('/Users/ansh/Proj_Res_3/notebook_02_hmm.ipynb', 'w') as f:
    nbf.write(nb_hmm, f)
print("Notebooks created.")
